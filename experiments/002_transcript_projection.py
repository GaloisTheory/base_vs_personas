# Experiment 002: Transcript Projection onto Assistant Axis
#
# Measure whether the "assistant personality" is more persistent in instruct-tuned
# OLMo models vs base/RL-zero across multi-turn conversations. We load cached
# persona vectors for 4 OLMo 3 7B variants, compute the assistant axis, then
# project each model's hidden states onto that axis turn-by-turn through a
# conversation transcript.
#
# See: experiments/plan/plan.md

# %% Cell 1: Imports & Config
import gc
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.persona_vectors_utils import compute_axes
from utils.transcript_projection import discover_transcripts, project_transcript

LAYER = 21  # 65% of 32 layers — matches cached persona vectors
MAX_SEQ_LEN = 8192  # Conservative; OLMo3 supports 4096+ natively

# Model variants: display_name -> (hf_id, slug matching cached .pt files)
MODELS = {
    "Base": ("allenai/Olmo-3-1025-7B", "Olmo-3-1025-7B"),
    "Instruct": ("allenai/Olmo-3-7B-Instruct", "Olmo-3-7B-Instruct"),
    "Think": ("allenai/Olmo-3-7B-Think", "Olmo-3-7B-Think"),
    "RL-Zero": ("allenai/Olmo-3-7B-RL-Zero-General", "Olmo-3-7B-RL-Zero-General"),
}

ASSISTANT_PERSONAS = ["default", "default_assistant", "default_llm", "default_helpful", "assistant"]

# When True, use Instruct model's axis for all variants (directly comparable).
# When False, each model uses its own axis.
SHARED_AXIS = True

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTORS_DIR = PROJECT_DIR / "data" / "persona_vectors"
TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "context" / "assistant-axis" / "transcripts"

HF_TOKEN = None  # Set if needed for gated models; OLMo3 is open

# %% Cell 2: Load Cached Persona Vectors → Compute Axes (CPU only)

# Load cached persona vectors for each variant
all_persona_vectors: dict[str, dict[str, torch.Tensor]] = {}
all_axes: dict[str, torch.Tensor] = {}

for name, (_, slug) in MODELS.items():
    path = VECTORS_DIR / f"olmo3_variants_{slug}_layer{LAYER}.pt"
    vectors = torch.load(path, map_location="cpu", weights_only=True)
    all_persona_vectors[name] = vectors
    axes_result = compute_axes(vectors, ASSISTANT_PERSONAS)
    all_axes[name] = axes_result["assistant_axis"]
    print(f"{name}: loaded {len(vectors)} persona vectors, axis norm={all_axes[name].norm():.4f}")

# If SHARED_AXIS, override all axes with Instruct's axis
if SHARED_AXIS:
    shared = all_axes["Instruct"]
    for name in all_axes:
        all_axes[name] = shared
    print("\nUsing SHARED axis (Instruct model's axis for all variants)")
else:
    print("\nUsing MODEL-SPECIFIC axes (each variant uses its own axis)")

# Sanity check: pairwise cosine similarity between axes (always informative)
print("\nAxis cosine similarities:")
variant_names = list(MODELS.keys())
# Compute from original per-model axes (not overridden shared ones)
original_axes = {
    name: compute_axes(all_persona_vectors[name], ASSISTANT_PERSONAS)["assistant_axis"]
    for name in variant_names
}
for a, b in combinations(variant_names, 2):
    cos = torch.dot(original_axes[a], original_axes[b]).item()
    print(f"  {a} vs {b}: {cos:.4f}")

# %% Cell 3: Load ALL Transcripts

transcripts: dict[str, list[dict]] = discover_transcripts(TRANSCRIPTS_DIR)

print(f"Loaded {len(transcripts)} transcripts:")
for label, conv in transcripts.items():
    n_asst = sum(1 for m in conv if m["role"] == "assistant")
    print(f"  {label}: {len(conv)} messages ({n_asst} assistant turns)")


# %% Cell 4: Run All Models × All Transcripts (load each model once)

SELECTED_TRANSCRIPT = "llama-70b/selfharm_unsteered"  # Used by Cells 5-7 for detailed views

all_results: dict[str, dict[str, np.ndarray]] = {}  # all_results[model_name][label]

for name, (hf_id, _) in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Loading: {name} ({hf_id})")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        token=HF_TOKEN,
    )

    all_results[name] = {}
    for label, conv in transcripts.items():
        print(f"\n  --- {label} ---")
        all_results[name][label] = project_transcript(
            model, tokenizer, conv, all_axes[name],
            layer=LAYER, max_seq_len=MAX_SEQ_LEN,
        )

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n  GPU memory freed for {name}")

# Backward-compat: results dict for the selected transcript (used by Cells 5-7)
results: dict[str, np.ndarray] = {
    name: all_results[name][SELECTED_TRANSCRIPT] for name in MODELS
}

print(f"\n{'='*60}")
print(f"All models × all transcripts processed!")
print(f"{'='*60}")
for name in MODELS:
    n_transcripts = len(all_results[name])
    sel = results[name]
    print(f"  {name}: {n_transcripts} transcripts | selected ({SELECTED_TRANSCRIPT}): {len(sel)} turns, range=[{sel.min():.4f}, {sel.max():.4f}]")


# %% Cell 5: Plot + Raw Values (Selected Transcript)

fig, ax = plt.subplots(figsize=(12, 6))

colors = {"Base": "#1f77b4", "Instruct": "#ff7f0e", "Think": "#2ca02c", "RL-Zero": "#d62728"}
markers = {"Base": "o", "Instruct": "s", "Think": "^", "RL-Zero": "D"}

for name, projs in results.items():
    turns = np.arange(1, len(projs) + 1)
    ax.plot(
        turns, projs,
        color=colors[name], marker=markers[name],
        label=name, linewidth=2, markersize=6,
    )

ax.set_xlabel("Assistant Turn", fontsize=12)
ax.set_ylabel("Projection onto Assistant Axis", fontsize=12)
axis_type = "Shared (Instruct)" if SHARED_AXIS else "Model-Specific"
ax.set_title(
    f"Assistant Axis Projection — {SELECTED_TRANSCRIPT}\n"
    f"(Layer {LAYER}, {axis_type} Axis)",
    fontsize=13,
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

plt.tight_layout()
(PROJECT_DIR / "images").mkdir(parents=True, exist_ok=True)
save_slug = SELECTED_TRANSCRIPT.replace("/", "_")
plt.savefig(
    PROJECT_DIR / "images" / f"transcript_projection_{save_slug}.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# Print raw values table
print(f"\nRaw projection values ({axis_type} axis, {SELECTED_TRANSCRIPT}):")
print(f"{'Turn':>5}", end="")
for name in results:
    print(f"  {name:>10}", end="")
print()
print("-" * (5 + 12 * len(results)))

max_turns = max(len(p) for p in results.values())
for t in range(max_turns):
    print(f"{t+1:>5}", end="")
    for name in results:
        projs = results[name]
        if t < len(projs):
            print(f"  {projs[t]:>10.4f}", end="")
        else:
            print(f"  {'---':>10}", end="")
    print()

# %% Cell 6: Chat Template Format — Instruct + Think (Selected Transcript)

CHAT_MODELS = {k: v for k, v in MODELS.items() if k in ("Instruct", "Think")}
conversation = transcripts[SELECTED_TRANSCRIPT]

results_chat: dict[str, np.ndarray] = {}

for name, (hf_id, _) in CHAT_MODELS.items():
    print(f"\n{'='*60}")
    print(f"Processing: {name} ({hf_id}) [chat template]")
    print(f"{'='*60}")

    print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=HF_TOKEN)
    print(f"  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        token=HF_TOKEN,
    )

    results_chat[name] = project_transcript(
        model, tokenizer, conversation, all_axes[name],
        format_mode="chat", layer=LAYER, max_seq_len=MAX_SEQ_LEN,
    )

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU memory freed")

print(f"\n{'='*60}")
print("Chat-format models processed!")
print(f"{'='*60}")
for name, projs in results_chat.items():
    print(f"  {name}: {len(projs)} turns, range=[{projs.min():.4f}, {projs.max():.4f}]")


# %% Cell 7: Comparison Plot — Raw vs Chat Template

fig, axes_plot = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

colors_fmt = {"raw": "#888888", "chat": "#ff7f0e"}

for idx, name in enumerate(CHAT_MODELS):
    ax = axes_plot[idx]
    raw_projs = results[name]
    chat_projs = results_chat[name]

    turns_raw = np.arange(1, len(raw_projs) + 1)
    turns_chat = np.arange(1, len(chat_projs) + 1)

    ax.plot(
        turns_raw, raw_projs,
        color=colors_fmt["raw"], marker="o", linestyle="--",
        label="Raw format", linewidth=2, markersize=6,
    )
    ax.plot(
        turns_chat, chat_projs,
        color=colors_fmt["chat"], marker="s", linestyle="-",
        label="Chat template", linewidth=2, markersize=6,
    )

    ax.set_xlabel("Assistant Turn", fontsize=12)
    if idx == 0:
        ax.set_ylabel("Projection onto Assistant Axis", fontsize=12)
    ax.set_title(f"{name}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

axis_type = "Shared (Instruct)" if SHARED_AXIS else "Model-Specific"
fig.suptitle(
    f"Raw vs Chat Template Format — {SELECTED_TRANSCRIPT}\n"
    f"(Layer {LAYER}, {axis_type} Axis)",
    fontsize=14,
)
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "images" / f"transcript_projection_{save_slug}_raw_vs_chat.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# Print raw values table
print(f"\nRaw vs Chat projection values ({axis_type} axis):")
for name in CHAT_MODELS:
    raw_projs = results[name]
    chat_projs = results_chat[name]
    print(f"\n  {name}:")
    print(f"  {'Turn':>5}  {'Raw':>10}  {'Chat':>10}  {'Diff':>10}")
    print(f"  {'-'*40}")
    max_turns = max(len(raw_projs), len(chat_projs))
    for t in range(max_turns):
        r = raw_projs[t] if t < len(raw_projs) else float("nan")
        c = chat_projs[t] if t < len(chat_projs) else float("nan")
        d = c - r if not (np.isnan(r) or np.isnan(c)) else float("nan")
        print(f"  {t+1:>5}  {r:>10.4f}  {c:>10.4f}  {d:>+10.4f}")

# %% Cell 8: All Transcripts — 4×4 Subplot Grid

labels = sorted(transcripts.keys())
n_plots = len(labels)
ncols = 4
nrows = (n_plots + ncols - 1) // ncols  # ceil division

fig, axes_grid = plt.subplots(nrows, ncols, figsize=(24, 5 * nrows), sharey=True)
axes_flat = axes_grid.flatten()

for idx, label in enumerate(labels):
    ax = axes_flat[idx]
    for name in MODELS:
        projs = all_results[name][label]
        turns = np.arange(1, len(projs) + 1)
        ax.plot(
            turns, projs,
            color=colors[name], marker=markers[name],
            linewidth=1.5, markersize=4,
        )
    ax.set_title(label, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    if idx >= (nrows - 1) * ncols:
        ax.set_xlabel("Turn", fontsize=9)

# Hide unused subplots
for idx in range(n_plots, len(axes_flat)):
    axes_flat[idx].set_visible(False)

# Shared legend at top
handles = [
    plt.Line2D([0], [0], color=colors[n], marker=markers[n], linewidth=2, markersize=6, label=n)
    for n in MODELS
]
fig.legend(handles=handles, loc="upper center", ncol=len(MODELS), fontsize=12,
           bbox_to_anchor=(0.5, 1.02))

axis_type = "Shared (Instruct)" if SHARED_AXIS else "Model-Specific"
fig.suptitle(
    f"Assistant Axis Projection — All Transcripts (Layer {LAYER}, {axis_type} Axis)",
    fontsize=15, y=1.05,
)
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "images" / "transcript_projection_all_transcripts.png",
    dpi=150, bbox_inches="tight",
)
plt.show()


# %% Cell 9: Mean Projection with Error Bars (First 10 Turns)

MAX_TURNS_MEAN = 10

fig, ax = plt.subplots(figsize=(10, 6))

print(f"Mean projection across {len(transcripts)} transcripts (first {MAX_TURNS_MEAN} turns):\n")
print(f"{'Model':<12} " + " ".join(f"{'T'+str(t+1):>8}" for t in range(MAX_TURNS_MEAN)))
print("-" * (12 + 9 * MAX_TURNS_MEAN))

for name in MODELS:
    # Collect first MAX_TURNS_MEAN turns from all transcripts, NaN-pad short ones
    padded = np.full((len(transcripts), MAX_TURNS_MEAN), np.nan)
    for i, label in enumerate(sorted(transcripts.keys())):
        projs = all_results[name][label]
        n = min(len(projs), MAX_TURNS_MEAN)
        padded[i, :n] = projs[:n]

    means = np.nanmean(padded, axis=0)
    counts = np.sum(~np.isnan(padded), axis=0)
    stds = np.nanstd(padded, axis=0, ddof=1)
    sems = stds / np.sqrt(counts)

    turns = np.arange(1, MAX_TURNS_MEAN + 1)
    ax.errorbar(
        turns, means, yerr=sems,
        color=colors[name], marker=markers[name],
        label=name, linewidth=2, markersize=6,
        capsize=3,
    )

    # Print summary row
    vals = " ".join(f"{m:>8.4f}" for m in means)
    print(f"{name:<12} {vals}")

print()
print(f"{'N transcripts':<12} " + " ".join(f"{int(c):>8}" for c in counts))

ax.set_xlabel("Assistant Turn", fontsize=12)
ax.set_ylabel("Mean Projection onto Assistant Axis", fontsize=12)
ax.set_xticks(range(1, MAX_TURNS_MEAN + 1))
axis_type = "Shared (Instruct)" if SHARED_AXIS else "Model-Specific"
ax.set_title(
    f"Mean Assistant Axis Projection ± SEM (First {MAX_TURNS_MEAN} Turns, N={len(transcripts)} transcripts)\n"
    f"(Layer {LAYER}, {axis_type} Axis)",
    fontsize=13,
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "images" / "transcript_projection_mean_first10.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# %%
