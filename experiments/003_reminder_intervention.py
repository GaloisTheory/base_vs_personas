# Experiment 003: Generation-Based System Prompt Intervention
#
# Previous version did forward passes on an existing transcript with different
# system prompt injection strategies. The assistant responses were always the
# original Llama-70b responses — we only measured how the static representation
# changed with added prompts.
#
# This version answers a different question: if OLMo actually *generates* its
# own responses, does a system prompt reminder prevent drift on the assistant axis?
#
# Two conditions:
#   1. Baseline         — Feed user messages, generate fresh assistant responses, no system prompt
#   2. + System prompt  — Same user messages, but prepend "You are a helpful assistant." as system message
#
# Plus: original transcript forward-pass projection as a reference line.

# %% Cell 1: Imports & Config
import gc
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.persona_vectors_utils import compute_axes
from utils.transcript_projection import discover_transcripts, project_transcript

LAYER = 21  # 65% of 32 layers — matches cached persona vectors
MAX_SEQ_LEN = 8192
MAX_NEW_TOKENS = 512

MODEL_HF_ID = "allenai/Olmo-3-7B-Instruct"
MODEL_SLUG = "Olmo-3-7B-Instruct"

ASSISTANT_PERSONAS = ["default", "default_assistant", "default_llm", "default_helpful", "assistant"]

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTORS_DIR = PROJECT_DIR / "data" / "persona_vectors"
TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "assistant-axis" / "transcripts"

HF_TOKEN = None  # OLMo is open

# %% Cell 2: Load Persona Vectors → Compute Axis

path = VECTORS_DIR / f"olmo3_variants_{MODEL_SLUG}_layer{LAYER}.pt"
vectors = torch.load(path, map_location="cpu", weights_only=True)
axes_result = compute_axes(vectors, ASSISTANT_PERSONAS)
assistant_axis = axes_result["assistant_axis"]

print(f"Loaded {len(vectors)} persona vectors from {path.name}")
print(f"Assistant axis norm: {assistant_axis.norm():.4f}")
print(f"Cosine(assistant_axis, PC1): {axes_result['cosine_sim']:.4f}")
print(f"PC1 variance explained: {axes_result['var_explained']:.2%}")

# %% Cell 3: Load Transcript, Extract User Messages

transcripts = discover_transcripts(TRANSCRIPTS_DIR)
SELECTED = "llama-70b/selfharm_unsteered"
conv_original = transcripts[SELECTED]

user_messages = [msg["content"] for msg in conv_original if msg["role"] == "user"]
n_user = len(user_messages)
n_asst = sum(1 for m in conv_original if m["role"] == "assistant")

print(f"Transcript: {SELECTED}")
print(f"  {len(conv_original)} messages, {n_user} user turns, {n_asst} assistant turns")
print(f"  First user message: {user_messages[0][:80]}...")

# %% Cell 4: generate_and_measure() function


def generate_and_measure(
    model,
    tokenizer,
    user_messages: list[str],
    axis: torch.Tensor,
    system_prompt: str | None = None,
    layer: int = LAYER,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_seq_len: int = MAX_SEQ_LEN,
) -> tuple[np.ndarray, list[dict]]:
    """Generate responses turn-by-turn, measuring projection at each turn.

    For each user message:
    1. Append user message to growing conversation
    2. Generate assistant response via model.generate() (greedy)
    3. Append generated response to conversation
    4. Forward pass on full conversation → extract hidden states at target layer
    5. Find token span for latest assistant response, mean-pool, project onto axis
    6. Record projection

    Returns:
        (projections_array, full_conversation)
    """
    conversation: list[dict] = []

    axis_device = axis.to(model.device)
    projections = []

    for turn_idx, user_msg in enumerate(user_messages):
        # 1. Append user message
        conversation.append({"role": "user", "content": user_msg})

        # Insert system reminder before every assistant generation
        if system_prompt is not None:
            conversation.append({"role": "system", "content": system_prompt})

        # 2. Generate assistant response (greedy decoding)
        gen_input_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        gen_inputs = tokenizer(gen_input_text, return_tensors="pt")
        gen_input_len = gen_inputs.input_ids.shape[1]

        # Truncate if too long before generation
        if gen_input_len > max_seq_len - max_new_tokens:
            print(f"  Turn {turn_idx}: WARNING — input {gen_input_len} tokens, "
                  f"truncating to {max_seq_len - max_new_tokens}")
            gen_inputs = {k: v[:, -(max_seq_len - max_new_tokens):] for k, v in gen_inputs.items()}
            gen_input_len = gen_inputs["input_ids"].shape[1]

        gen_inputs = {k: v.to(model.device) for k, v in gen_inputs.items()}

        with torch.inference_mode():
            output_ids = model.generate(
                **gen_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode only new tokens
        new_token_ids = output_ids[0, gen_input_len:]
        response_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

        # 3. Append generated response to conversation
        conversation.append({"role": "assistant", "content": response_text})

        # 4. Forward pass on full conversation for hidden states
        full_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        full_inputs = tokenizer(full_text, return_tensors="pt")
        full_seq_len = full_inputs.input_ids.shape[1]

        if full_seq_len > max_seq_len:
            print(f"  Turn {turn_idx}: WARNING — full seq {full_seq_len} tokens, "
                  f"truncating to {max_seq_len}")
            full_inputs = {k: v[:, :max_seq_len] for k, v in full_inputs.items()}
            full_seq_len = max_seq_len

        full_inputs = {k: v.to(model.device) for k, v in full_inputs.items()}

        with torch.inference_mode():
            outputs = model(**full_inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states[layer][0].float()  # (seq_len, d_model)

        # 5. Find token span for the LAST assistant response
        # Prefix = conversation without last assistant message, with generation prompt
        prefix_text = tokenizer.apply_chat_template(
            conversation[:-1], tokenize=False, add_generation_prompt=True
        ).rstrip()
        prefix_tokens = tokenizer(prefix_text, return_tensors="pt").input_ids.shape[1]

        response_start = prefix_tokens + 1  # +1 for space/token after gen prompt
        response_end = full_seq_len

        # Clamp to valid range
        response_start = min(response_start, full_seq_len)
        response_end = min(response_end, full_seq_len)

        if response_start >= response_end:
            print(f"  Turn {turn_idx}: WARNING — empty span [{response_start}:{response_end}], "
                  f"using last token")
            response_start = response_end - 1

        span_hidden = hidden_states[response_start:response_end]
        mean_activation = span_hidden.mean(dim=0)
        proj = torch.dot(mean_activation, axis_device).item()
        projections.append(proj)

        # Free intermediate tensors
        del outputs, hidden_states, full_inputs, gen_inputs, output_ids
        torch.cuda.empty_cache()

        resp_preview = response_text[:60].replace("\n", " ")
        print(f"  Turn {turn_idx}: tokens [{response_start}:{response_end}] "
              f"({response_end - response_start} tok), proj={proj:.4f}  "
              f'"{resp_preview}..."')

    return np.array(projections), conversation


# %% Cell 5: Run Experiment

print(f"Loading model: {MODEL_HF_ID}")
tok = AutoTokenizer.from_pretrained(MODEL_HF_ID, token=HF_TOKEN)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_HF_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
    token=HF_TOKEN,
)

# --- Condition 1: Baseline (no system prompt) ---
print(f"\n{'='*60}")
print("Condition: Baseline (no system prompt)")
print(f"{'='*60}")
proj_baseline, conv_baseline = generate_and_measure(
    mdl, tok, user_messages, assistant_axis,
    system_prompt=None,
)

# --- Condition 2: + System prompt ---
print(f"\n{'='*60}")
print("Condition: + System prompt")
print(f"{'='*60}")
proj_sysprompt, conv_sysprompt = generate_and_measure(
    mdl, tok, user_messages, assistant_axis,
    system_prompt="Reminder: You are a helpful assistant.",
)

# --- Reference: Original transcript forward-pass projection ---
print(f"\n{'='*60}")
print("Reference: Original transcript (forward-pass only)")
print(f"{'='*60}")
proj_reference = project_transcript(
    mdl, tok, conv_original, assistant_axis,
    format_mode="chat", layer=LAYER, max_seq_len=MAX_SEQ_LEN,
)

# Print first and last response per condition for sanity check
for label, conv in [("Baseline", conv_baseline), ("+ System prompt", conv_sysprompt)]:
    asst_msgs = [m for m in conv if m["role"] == "assistant"]
    print(f"\n--- {label}: First response ---")
    print(asst_msgs[0]["content"][:200])
    print(f"\n--- {label}: Last response ---")
    print(asst_msgs[-1]["content"][:200])

# Save generated conversations
GENERATED_DIR = PROJECT_DIR / "data" / "generated_transcripts"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

for label, conv, projs in [
    ("003_baseline", conv_baseline, proj_baseline),
    ("003_sysprompt", conv_sysprompt, proj_sysprompt),
]:
    out_path = GENERATED_DIR / f"{label}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL_HF_ID,
            "transcript_source": SELECTED,
            "condition": label,
            "projections": projs.tolist(),
            "conversation": conv,
        }, f, indent=2)
    print(f"Saved {out_path}")

del mdl, tok
gc.collect()
torch.cuda.empty_cache()
print("\nGPU memory freed")

# %% Cell 6: Plot + Raw Values

fig, ax = plt.subplots(figsize=(12, 6))

n_turns_baseline = len(proj_baseline)
n_turns_sysprompt = len(proj_sysprompt)
n_turns_reference = len(proj_reference)

# Reference: original transcript (dashed gray)
turns_ref = np.arange(1, n_turns_reference + 1)
ax.plot(
    turns_ref, proj_reference,
    color="gray", linestyle="--", marker=".",
    label="Original transcript (reference)", linewidth=1.5, markersize=5, alpha=0.7,
)

# Generated baseline (blue)
turns_base = np.arange(1, n_turns_baseline + 1)
ax.plot(
    turns_base, proj_baseline,
    color="#1f77b4", linestyle="-", marker="o",
    label="Generated baseline (no sys prompt)", linewidth=2, markersize=6,
)

# Generated + system prompt (orange)
turns_sys = np.arange(1, n_turns_sysprompt + 1)
ax.plot(
    turns_sys, proj_sysprompt,
    color="#ff7f0e", linestyle="-", marker="s",
    label='Generated + "You are a helpful assistant."', linewidth=2, markersize=6,
)

ax.set_xlabel("Assistant Turn", fontsize=12)
ax.set_ylabel("Projection onto Assistant Axis", fontsize=12)
ax.set_title(
    f"Generation-Based Intervention — {SELECTED}\n"
    f"(OLMo 3 7B Instruct, Layer {LAYER}, Greedy Decoding)",
    fontsize=13,
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

plt.tight_layout()
(PROJECT_DIR / "images").mkdir(parents=True, exist_ok=True)
plt.savefig(
    PROJECT_DIR / "images" / "003_reminder_intervention.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# --- Raw values table ---
print(f"\nRaw projection values ({SELECTED}):")
header = f"{'Turn':>5}  {'Reference':>12}  {'Baseline':>12}  {'+ SysPrompt':>12}  {'Diff (sys-base)':>16}"
print(header)
print("-" * len(header))

max_turns = max(n_turns_baseline, n_turns_sysprompt, n_turns_reference)
for t in range(max_turns):
    row = f"{t+1:>5}"
    row += f"  {proj_reference[t]:>12.4f}" if t < n_turns_reference else f"  {'---':>12}"
    row += f"  {proj_baseline[t]:>12.4f}" if t < n_turns_baseline else f"  {'---':>12}"
    row += f"  {proj_sysprompt[t]:>12.4f}" if t < n_turns_sysprompt else f"  {'---':>12}"
    if t < n_turns_baseline and t < n_turns_sysprompt:
        diff = proj_sysprompt[t] - proj_baseline[t]
        row += f"  {diff:>+16.4f}"
    else:
        row += f"  {'---':>16}"
    print(row)

# Summary stats
n_common = min(n_turns_baseline, n_turns_sysprompt)
mean_diff = np.mean(proj_sysprompt[:n_common] - proj_baseline[:n_common])
print(f"\nMean diff (system prompt - baseline): {mean_diff:+.4f}")
print(f"Baseline mean projection: {np.mean(proj_baseline):+.4f}")
print(f"System prompt mean projection: {np.mean(proj_sysprompt):+.4f}")
print(f"Reference mean projection: {np.mean(proj_reference):+.4f}")

# %%
conv_sysprompt
# %%
