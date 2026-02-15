# Experiment 004: Base vs Instruct Batched Generation
#
# Follow-up to 003b — instead of varying the system prompt, we vary the model
# (Base vs Instruct) while keeping format constant (raw plaintext). This tests
# whether the model variant matters more than the system prompt for maintaining
# "assistant-ness" during selfharm transcript generation.
#
# Two generation conditions (both raw format, no system prompt):
#   1. base_raw     — OLMo-3-1025-7B (base model)
#   2. instruct_raw — OLMo-3-7B-Instruct (instruct model)
#
# Plus: 003b baseline reference (Instruct, chat template) loaded from JSON.
#
# Shared axis: Instruct model's assistant axis (same as 002's SHARED_AXIS=True).
# Generation is micro-batched (4 at a time) for speed; measurement is
# micro-batched (4 at a time) to avoid OOM from output_hidden_states.

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
from utils.transcript_projection import (
    discover_transcripts,
    format_conversation_raw,
)

LAYER = 21  # 65% of 32 layers — matches cached persona vectors
MAX_SEQ_LEN = 8192
MAX_NEW_TOKENS = 512

# Model variants: display_name -> (hf_id, slug matching cached .pt files)
MODELS = {
    "Base": ("allenai/Olmo-3-1025-7B", "Olmo-3-1025-7B"),
    "Instruct": ("allenai/Olmo-3-7B-Instruct", "Olmo-3-7B-Instruct"),
}

ASSISTANT_PERSONAS = ["default", "default_assistant", "default_llm", "default_helpful", "assistant"]

N_SAMPLES = 16
TEMPERATURE = 0.5
SEED = 42
GEN_BATCH_SIZE = 4      # micro-batch for generation (avoids OOM on long conversations)
MEASURE_BATCH_SIZE = 4  # micro-batch for measurement (avoids OOM from output_hidden_states)

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTORS_DIR = PROJECT_DIR / "data" / "persona_vectors"
TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "assistant-axis" / "transcripts"
GENERATED_DIR = PROJECT_DIR / "data" / "generated_transcripts"

HF_TOKEN = None  # OLMo is open

# %% Cell 2: Load Shared Axis (Instruct model's persona vectors)

instruct_slug = MODELS["Instruct"][1]
path = VECTORS_DIR / f"olmo3_variants_{instruct_slug}_layer{LAYER}.pt"
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

# %% Cell 4: generate_and_measure_batched_raw() function


def generate_and_measure_batched_raw(
    model,
    tokenizer,
    user_messages: list[str],
    axis: torch.Tensor,
    n_samples: int = N_SAMPLES,
    layer: int = LAYER,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_seq_len: int = MAX_SEQ_LEN,
    temperature: float = TEMPERATURE,
    seed: int = SEED,
) -> tuple[np.ndarray, list[list[dict]]]:
    """Generate responses turn-by-turn for n_samples in parallel using raw format.

    Adapted from 003b's generate_and_measure_batched() with these changes:
    - Uses raw plaintext format instead of chat template
    - Adds stop-string truncation for base model (cuts at "User:", "System:")
    - No system prompt injection

    For each user message:
    1. Append user message to all n_samples conversations
    2. Batched generation — raw format prompt ending with "Assistant:"
    3. Post-process: truncate at turn boundaries (base model may continue)
    4. Append decoded responses to each sample's conversation
    5. Micro-batched measurement — forward pass, extract hidden states, project

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
        # --- 1. Append user message to all conversations ---
        for s in range(n_samples):
            conversations[s].append({"role": "user", "content": user_msg})

        # --- 2. Micro-batched generation (raw format) ---
        gen_texts = []
        for s in range(n_samples):
            # Build raw format prompt ending with "Assistant:" for generation
            gen_text = format_conversation_raw(conversations[s]) + "\n\nAssistant:"
            gen_texts.append(gen_text)

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

                # Post-process: truncate at first turn boundary
                # Base model may generate "User:" continuations — truncate
                for stop in ["\nUser:", "\n\nUser:", "\nSystem:", "\n\nSystem:"]:
                    idx = resp.find(stop)
                    if idx != -1:
                        resp = resp[:idx].strip()

                responses.append(resp)

            del output_ids, gen_inputs
            torch.cuda.empty_cache()

        # --- 3. Append responses to conversations ---
        for s in range(n_samples):
            conversations[s].append({"role": "assistant", "content": responses[s]})

        # --- 4. Micro-batched measurement (raw format) ---
        tokenizer.padding_side = "right"

        for mb_start in range(0, n_samples, MEASURE_BATCH_SIZE):
            mb_end = min(mb_start + MEASURE_BATCH_SIZE, n_samples)
            mb_size = mb_end - mb_start

            # Tokenize full conversations for this micro-batch (raw format)
            full_texts = []
            for s in range(mb_start, mb_end):
                full_text = format_conversation_raw(conversations[s])
                full_texts.append(full_text)

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

                # Prefix = conversation without last assistant message, formatted raw + "Assistant:"
                prefix_msgs = conversations[s][:-1]  # everything except last assistant response
                prefix_text = format_conversation_raw(prefix_msgs) + "\n\nAssistant:"
                prefix_tokens = tokenizer(prefix_text, return_tensors="pt").input_ids.shape[1]

                # +1 for space after colon (same offset as _find_assistant_spans_raw)
                response_start = prefix_tokens + 1

                # Attention mask tells us where real tokens are (not padding)
                attn_mask = full_inputs["attention_mask"][i]
                real_len = attn_mask.sum().item()

                response_start = min(response_start, real_len)
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
        print(f"  Turn {turn_idx:2d}: mean_proj={mean_proj:.4f} +/- {std_proj:.4f}  "
              f'"{resp_preview}..."')

    return projections, conversations


# %% Cell 5: Outer Model Loop — Load Each Model, Generate, Save, Free GPU

results = {}  # model_label -> (projections, conversations)

for model_name, (hf_id, slug) in MODELS.items():
    label = f"{model_name.lower()}_raw"

    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({hf_id})")
    print(f"Label: {label} ({N_SAMPLES} samples, T={TEMPERATURE})")
    print(f"{'='*60}")

    tok = AutoTokenizer.from_pretrained(hf_id, token=HF_TOKEN)
    tok.pad_token = tok.eos_token  # OLMo doesn't set pad_token by default

    mdl = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        token=HF_TOKEN,
    )

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    projs, convs = generate_and_measure_batched_raw(
        mdl, tok, user_messages, assistant_axis,
        n_samples=N_SAMPLES,
        seed=SEED,
    )
    results[label] = (projs, convs)

    # Save JSON
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GENERATED_DIR / f"004_{label}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": hf_id,
            "transcript_source": SELECTED,
            "condition": f"004_{label}",
            "n_samples": N_SAMPLES,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "format": "raw",
            "system_prompt": None,
            "projections": projs.tolist(),  # (n_samples, n_turns)
            "conversations": convs,  # list of n_samples conversation lists
        }, f, indent=2)
    print(f"Saved {out_path}")

    # Print first response per sample for sanity check
    asst_0 = [m for m in convs[0] if m["role"] == "assistant"]
    asst_1 = [m for m in convs[1] if m["role"] == "assistant"]
    print(f"\n--- {label} sample 0, turn 0 ---")
    print(asst_0[0]["content"][:200])
    print(f"--- {label} sample 1, turn 0 ---")
    print(asst_1[0]["content"][:200])

    # Free GPU before loading next model
    del mdl, tok
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nGPU memory freed for {model_name}")

# %% Cell 6: Load 003b Baseline Reference

ref_path = GENERATED_DIR / "003b_baseline.json"
with open(ref_path) as f:
    ref_data = json.load(f)

proj_003b_ref = np.array(ref_data["projections"])  # (n_samples, n_turns)
print(f"Loaded 003b baseline reference from {ref_path.name}")
print(f"  Shape: {proj_003b_ref.shape} (samples x turns)")
print(f"  Mean projection: {proj_003b_ref.mean():.4f}")

# %% Cell 7: Plot + Summary Stats

fig, ax = plt.subplots(figsize=(14, 7))

# Colors and labels for generation conditions
COLORS = {
    "base_raw": "#1f77b4",       # blue
    "instruct_raw": "#ff7f0e",   # orange
    "003b_ref": "#888888",       # gray
}
LABELS = {
    "base_raw": "Base (raw format)",
    "instruct_raw": "Instruct (raw format)",
    "003b_ref": "003b Baseline ref (Instruct, chat)",
}
MARKERS = {
    "base_raw": "o",
    "instruct_raw": "s",
    "003b_ref": ".",
}

# --- 003b reference (dashed gray) ---
ref_mean = proj_003b_ref.mean(axis=0)
ref_sem = proj_003b_ref.std(axis=0) / np.sqrt(proj_003b_ref.shape[0])
ref_turns = np.arange(1, len(ref_mean) + 1)

ax.fill_between(ref_turns, ref_mean - ref_sem, ref_mean + ref_sem,
                color=COLORS["003b_ref"], alpha=0.15)
ax.plot(
    ref_turns, ref_mean,
    color=COLORS["003b_ref"], linestyle="--", marker=MARKERS["003b_ref"],
    label=LABELS["003b_ref"], linewidth=1.5, markersize=5, alpha=0.7,
)

# --- Generation conditions ---
for label in ["base_raw", "instruct_raw"]:
    projs, _ = results[label]
    n_turns = projs.shape[1]
    t = np.arange(1, n_turns + 1)

    mean = projs.mean(axis=0)
    sem = projs.std(axis=0) / np.sqrt(N_SAMPLES)
    color = COLORS[label]

    # Individual traces (thin, translucent)
    for s in range(N_SAMPLES):
        ax.plot(t, projs[s], color=color, alpha=0.1, linewidth=0.5)

    # Shaded SEM band
    ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.2)

    # Mean line with markers
    ax.plot(
        t, mean,
        color=color, linestyle="-", marker=MARKERS[label],
        label=LABELS[label], linewidth=2, markersize=6,
    )

    # Error bars (SEM)
    ax.errorbar(
        t, mean, yerr=sem,
        color=color, fmt="none", capsize=3, capthick=1, linewidth=1, alpha=0.7,
    )

ax.set_xlabel("Assistant Turn", fontsize=12)
ax.set_ylabel("Projection onto Assistant Axis", fontsize=12)
ax.set_title(
    f"Base vs Instruct Batched Generation — {SELECTED}\n"
    f"(OLMo 3 7B, Layer {LAYER}, T={TEMPERATURE}, {N_SAMPLES} samples, raw format, shared Instruct axis)",
    fontsize=13,
)
ax.legend(fontsize=10, loc="best")
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

plt.tight_layout()
(PROJECT_DIR / "images").mkdir(parents=True, exist_ok=True)
plt.savefig(
    PROJECT_DIR / "images" / "004_base_vs_instruct_batched.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# --- Raw values table (mean +/- SEM per turn per condition) ---
gen_labels = ["base_raw", "instruct_raw"]
print(f"\nProjection values — mean +/- SEM ({SELECTED}, {N_SAMPLES} samples):")
header = f"{'Turn':>5}  {'003b ref':>18}"
for label in gen_labels:
    header += f"  {label:>24}"
print(header)
print("-" * len(header))

n_ref_turns = len(ref_mean)
max_turns = max(n_ref_turns, *(results[l][0].shape[1] for l in gen_labels))
for t in range(max_turns):
    row = f"{t+1:>5}"
    if t < n_ref_turns:
        row += f"  {ref_mean[t]:>8.4f} +/- {ref_sem[t]:<6.4f}"
    else:
        row += f"  {'---':>18}"
    for label in gen_labels:
        projs, _ = results[label]
        if t < projs.shape[1]:
            m = projs[:, t].mean()
            s = projs[:, t].std() / np.sqrt(N_SAMPLES)
            row += f"  {m:>10.4f} +/- {s:<8.4f}"
        else:
            row += f"  {'---':>24}"
    print(row)

# --- Summary stats ---
print(f"\nSummary (mean projection across all turns):")
print(f"  003b reference:    {ref_mean.mean():+.4f} +/- {ref_mean.std() / np.sqrt(len(ref_mean)):.4f}")
for label in gen_labels:
    projs, _ = results[label]
    grand_mean = projs.mean()
    grand_sem = projs.mean(axis=1).std() / np.sqrt(N_SAMPLES)
    print(f"  {label:20s}: {grand_mean:+.4f} +/- {grand_sem:.4f}")

# Pairwise diffs
for i in range(len(gen_labels)):
    for j in range(i + 1, len(gen_labels)):
        a_label, b_label = gen_labels[i], gen_labels[j]
        a_means = results[a_label][0].mean(axis=1)  # (n_samples,)
        b_means = results[b_label][0].mean(axis=1)  # (n_samples,)
        diff = b_means.mean() - a_means.mean()
        print(f"  Diff ({b_label} - {a_label}): {diff:+.4f}")

# %%
