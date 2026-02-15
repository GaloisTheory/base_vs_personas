# Experiment 003b: Batched Generation with Temperature Sampling
#
# Extension of 003 — instead of 1 greedy rollout per condition, we generate
# 16 independent sampled rollouts (temperature=0.5) to get variance estimates.
#
# Three conditions:
#   1. Baseline         — No system prompt
#   2. Simple sysprompt — "Reminder: You are a helpful assistant."
#   3. Complex sysprompt — Detailed guidelines (tone, clarity, honesty, context)
#
# Plus: original transcript forward-pass projection as a reference line.
#
# Generation is batched (16 simultaneous) for speed on H100.
# Measurement is micro-batched (4 at a time) to avoid OOM from output_hidden_states.

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

N_SAMPLES = 16
TEMPERATURE = 0.5
SEED = 42
GEN_BATCH_SIZE = 4      # micro-batch for generation (avoids OOM on long conversations)
MEASURE_BATCH_SIZE = 4  # micro-batch for measurement (avoids OOM from output_hidden_states)

SYSTEM_PROMPT_SIMPLE = "Reminder: You are a helpful assistant."
SYSTEM_PROMPT_COMPLEX = (
    "You are a helpful, harmless, and honest AI assistant. Your goal is to provide "
    "accurate and relevant information to assist the user. Guidelines: Tone: Maintain "
    "a polite, professional, and approachable tone at all times. Clarity: Explain complex "
    "concepts simply and clearly. Avoid unnecessary jargon. Honesty: If you do not know "
    "the answer to a question, admit it gracefully rather than hallucinating information. "
    "Context: Always take into account the user's previous questions to provide a cohesive "
    "conversation."
)

CONDITIONS = [
    ("baseline", None),
    ("simple_sysprompt", SYSTEM_PROMPT_SIMPLE),
    ("complex_sysprompt", SYSTEM_PROMPT_COMPLEX),
]

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

# %% Cell 4: generate_and_measure_batched() function


def generate_and_measure_batched(
    model,
    tokenizer,
    user_messages: list[str],
    axis: torch.Tensor,
    n_samples: int = N_SAMPLES,
    system_prompt: str | None = None,
    layer: int = LAYER,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_seq_len: int = MAX_SEQ_LEN,
    temperature: float = TEMPERATURE,
    seed: int = SEED,
) -> tuple[np.ndarray, list[list[dict]]]:
    """Generate responses turn-by-turn for n_samples in parallel, measuring projection.

    For each user message (18 turns):
    1. Append user message (+ optional system prompt) to all n_samples conversations
    2. Batched generation — left-pad tokenize n_samples prompts, model.generate(temperature)
    3. Append decoded responses to each sample's conversation
    4. Micro-batched measurement — forward pass in groups of MEASURE_BATCH_SIZE,
       extract hidden states, find response span, mean-pool, project onto axis
    5. Cleanup GPU memory

    Returns:
        projections: (n_samples, n_turns) array
        conversations: list of n_samples conversation lists
    """
    axis_device = axis.to(model.device)
    n_turns = len(user_messages)
    projections = np.zeros((n_samples, n_turns))

    # Initialize n_samples independent conversation histories
    conversations: list[list[dict]] = [[] for _ in range(n_samples)]

    for turn_idx, user_msg in enumerate(user_messages):
        # --- 1. Append user message (+ system prompt) to all conversations ---
        for s in range(n_samples):
            conversations[s].append({"role": "user", "content": user_msg})
            if system_prompt is not None:
                conversations[s].append({"role": "system", "content": system_prompt})

        # --- 2. Micro-batched generation ---
        # Build generation prompts for all samples
        gen_texts = []
        for s in range(n_samples):
            text = tokenizer.apply_chat_template(
                conversations[s], tokenize=False, add_generation_prompt=True
            )
            gen_texts.append(text)

        tokenizer.padding_side = "left"
        responses = []

        for gb_start in range(0, n_samples, GEN_BATCH_SIZE):
            gb_end = min(gb_start + GEN_BATCH_SIZE, n_samples)
            batch_texts = gen_texts[gb_start:gb_end]

            gen_inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True,
                max_length=max_seq_len - max_new_tokens,
            )
            gen_input_len = gen_inputs.input_ids.shape[1]
            gen_inputs = {k: v.to(model.device) for k, v in gen_inputs.items()}

            # Set seed per micro-batch for reproducibility
            torch.manual_seed(seed + turn_idx * 1000 + gb_start)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + turn_idx * 1000 + gb_start)

            with torch.inference_mode():
                output_ids = model.generate(
                    **gen_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                )

            for i in range(gb_end - gb_start):
                new_tokens = output_ids[i, gen_input_len:]
                resp = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                responses.append(resp)

            del output_ids, gen_inputs
            torch.cuda.empty_cache()

        # --- 3. Append responses to conversations ---
        for s in range(n_samples):
            conversations[s].append({"role": "assistant", "content": responses[s]})

        # --- 4. Micro-batched measurement ---
        # Right-pad for forward pass (need attention mask to ignore padding)
        tokenizer.padding_side = "right"

        for mb_start in range(0, n_samples, MEASURE_BATCH_SIZE):
            mb_end = min(mb_start + MEASURE_BATCH_SIZE, n_samples)
            mb_size = mb_end - mb_start

            # Tokenize full conversations for this micro-batch
            full_texts = []
            for s in range(mb_start, mb_end):
                text = tokenizer.apply_chat_template(
                    conversations[s], tokenize=False, add_generation_prompt=False
                )
                full_texts.append(text)

            full_inputs = tokenizer(
                full_texts, return_tensors="pt", padding=True, truncation=True,
                max_length=max_seq_len,
            )
            full_inputs = {k: v.to(model.device) for k, v in full_inputs.items()}

            with torch.inference_mode():
                outputs = model(**full_inputs, output_hidden_states=True)

            hidden = outputs.hidden_states[layer].float()  # (mb_size, seq_len, d_model)

            # Per-sample: find response span, mean-pool, project
            for i in range(mb_size):
                s = mb_start + i

                # Prefix = conversation without last assistant message + gen prompt
                prefix_text = tokenizer.apply_chat_template(
                    conversations[s][:-1], tokenize=False, add_generation_prompt=True
                ).rstrip()
                prefix_tokens = tokenizer(prefix_text, return_tensors="pt").input_ids.shape[1]

                # Attention mask tells us where real tokens are (not padding)
                attn_mask = full_inputs["attention_mask"][i]
                real_len = attn_mask.sum().item()

                response_start = min(prefix_tokens + 1, real_len)
                response_end = real_len

                if response_start >= response_end:
                    response_start = response_end - 1

                span_hidden = hidden[i, response_start:response_end]
                mean_activation = span_hidden.mean(dim=0)
                proj = torch.dot(mean_activation, axis_device).item()
                projections[s, turn_idx] = proj

            del outputs, hidden, full_inputs
            torch.cuda.empty_cache()

        # Print progress
        mean_proj = projections[:, turn_idx].mean()
        std_proj = projections[:, turn_idx].std()
        resp_preview = responses[0][:50].replace("\n", " ")
        print(f"  Turn {turn_idx:2d}: mean_proj={mean_proj:.4f} ± {std_proj:.4f}  "
              f'"{resp_preview}..."')

    return projections, conversations


# %% Cell 5: Run Experiment

print(f"Loading model: {MODEL_HF_ID}")
tok = AutoTokenizer.from_pretrained(MODEL_HF_ID, token=HF_TOKEN)
tok.pad_token = tok.eos_token  # OLMo doesn't set pad_token by default
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_HF_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
    token=HF_TOKEN,
)

results = {}  # label -> (projections, conversations)

for cond_label, cond_prompt in CONDITIONS:
    print(f"\n{'='*60}")
    print(f"Condition: {cond_label} ({N_SAMPLES} samples, T={TEMPERATURE})")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    projs, convs = generate_and_measure_batched(
        mdl, tok, user_messages, assistant_axis,
        n_samples=N_SAMPLES,
        system_prompt=cond_prompt,
        seed=SEED,
    )
    results[cond_label] = (projs, convs)

# --- Reference: Original transcript forward-pass projection ---
print(f"\n{'='*60}")
print("Reference: Original transcript (forward-pass only)")
print(f"{'='*60}")
proj_reference = project_transcript(
    mdl, tok, conv_original, assistant_axis,
    format_mode="chat", layer=LAYER, max_seq_len=MAX_SEQ_LEN,
)

# Print first response per condition/sample for sanity check
for cond_label, _ in CONDITIONS:
    projs, convs = results[cond_label]
    asst_0 = [m for m in convs[0] if m["role"] == "assistant"]
    asst_1 = [m for m in convs[1] if m["role"] == "assistant"]
    print(f"\n--- {cond_label} sample 0, turn 0 ---")
    print(asst_0[0]["content"][:150])
    print(f"--- {cond_label} sample 1, turn 0 ---")
    print(asst_1[0]["content"][:150])

# Save results
GENERATED_DIR = PROJECT_DIR / "data" / "generated_transcripts"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

for cond_label, _ in CONDITIONS:
    projs, convs = results[cond_label]
    out_path = GENERATED_DIR / f"003b_{cond_label}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL_HF_ID,
            "transcript_source": SELECTED,
            "condition": f"003b_{cond_label}",
            "n_samples": N_SAMPLES,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "system_prompt": dict(CONDITIONS)[cond_label],
            "projections": projs.tolist(),  # (n_samples, n_turns)
            "conversations": convs,  # list of n_samples conversation lists
        }, f, indent=2)
    print(f"Saved {out_path}")

del mdl, tok
gc.collect()
torch.cuda.empty_cache()
print("\nGPU memory freed")

# %% Cell 6: Plot + Raw Values

fig, ax = plt.subplots(figsize=(14, 7))

n_turns_ref = len(proj_reference)
turns = np.arange(1, n_turns_ref + 1)

# Reference: original transcript (dashed gray)
ax.plot(
    turns, proj_reference,
    color="gray", linestyle="--", marker=".",
    label="Original transcript (reference)", linewidth=1.5, markersize=5, alpha=0.7,
)

# Colors for conditions
COLORS = {
    "baseline": "#1f77b4",          # blue
    "simple_sysprompt": "#ff7f0e",  # orange
    "complex_sysprompt": "#2ca02c", # green
}
LABELS = {
    "baseline": "Baseline (no sys prompt)",
    "simple_sysprompt": '+ Simple: "You are a helpful assistant."',
    "complex_sysprompt": "+ Complex: detailed guidelines",
}
MARKERS = {
    "baseline": "o",
    "simple_sysprompt": "s",
    "complex_sysprompt": "^",
}

for cond_label, _ in CONDITIONS:
    projs, _ = results[cond_label]
    n_turns = projs.shape[1]
    t = np.arange(1, n_turns + 1)

    mean = projs.mean(axis=0)
    sem = projs.std(axis=0) / np.sqrt(N_SAMPLES)
    color = COLORS[cond_label]

    # Individual traces (thin, translucent)
    for s in range(N_SAMPLES):
        ax.plot(t, projs[s], color=color, alpha=0.1, linewidth=0.5)

    # Shaded SEM band
    ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.2)

    # Mean line with markers
    ax.plot(
        t, mean,
        color=color, linestyle="-", marker=MARKERS[cond_label],
        label=LABELS[cond_label], linewidth=2, markersize=6,
    )

    # Error bars (SEM)
    ax.errorbar(
        t, mean, yerr=sem,
        color=color, fmt="none", capsize=3, capthick=1, linewidth=1, alpha=0.7,
    )

ax.set_xlabel("Assistant Turn", fontsize=12)
ax.set_ylabel("Projection onto Assistant Axis", fontsize=12)
ax.set_title(
    f"Batched Generation Intervention — {SELECTED}\n"
    f"(OLMo 3 7B Instruct, Layer {LAYER}, T={TEMPERATURE}, {N_SAMPLES} samples/condition)",
    fontsize=13,
)
ax.legend(fontsize=10, loc="best")
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

plt.tight_layout()
(PROJECT_DIR / "images").mkdir(parents=True, exist_ok=True)
plt.savefig(
    PROJECT_DIR / "images" / "003b_batched_reminder.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# --- Raw values table (mean ± SEM per turn per condition) ---
print(f"\nProjection values — mean ± SEM ({SELECTED}, {N_SAMPLES} samples/condition):")
header = f"{'Turn':>5}  {'Reference':>12}"
for cond_label, _ in CONDITIONS:
    header += f"  {cond_label:>24}"
print(header)
print("-" * len(header))

max_turns = max(n_turns_ref, *(results[c][0].shape[1] for c, _ in CONDITIONS))
for t in range(max_turns):
    row = f"{t+1:>5}"
    row += f"  {proj_reference[t]:>12.4f}" if t < n_turns_ref else f"  {'---':>12}"
    for cond_label, _ in CONDITIONS:
        projs, _ = results[cond_label]
        if t < projs.shape[1]:
            m = projs[:, t].mean()
            s = projs[:, t].std() / np.sqrt(N_SAMPLES)
            row += f"  {m:>10.4f} ± {s:<8.4f}"
        else:
            row += f"  {'---':>24}"
    print(row)

# --- Summary stats ---
print(f"\nSummary (mean projection across all turns):")
print(f"  Reference:         {np.mean(proj_reference):+.4f}")
for cond_label, _ in CONDITIONS:
    projs, _ = results[cond_label]
    grand_mean = projs.mean()
    grand_sem = projs.mean(axis=1).std() / np.sqrt(N_SAMPLES)
    print(f"  {cond_label:20s}: {grand_mean:+.4f} ± {grand_sem:.4f}")

# Pairwise diffs
cond_labels = [c for c, _ in CONDITIONS]
for i in range(len(cond_labels)):
    for j in range(i + 1, len(cond_labels)):
        a_label, b_label = cond_labels[i], cond_labels[j]
        a_means = results[a_label][0].mean(axis=1)  # (n_samples,)
        b_means = results[b_label][0].mean(axis=1)  # (n_samples,)
        diff = b_means.mean() - a_means.mean()
        print(f"  Diff ({b_label} - {a_label}): {diff:+.4f}")

# %%
