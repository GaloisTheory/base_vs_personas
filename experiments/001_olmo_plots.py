# %% [markdown]
# # Assistant Axis vs PC1 — OLMo 3 7B Variants
#
# Loads persona vectors extracted in experiment 001 and checks:
# **how aligned is the assistant axis with PC1?**

# %% Imports & config
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path("..").resolve()))
from utils.persona_vectors import load_persona_vectors
from utils.persona_vectors_utils import compute_axes

PROJECT_DIR = Path("..").resolve()
VECTORS_DIR = PROJECT_DIR / "data" / "persona_vectors"

# ---- Easy to edit ----
ASSISTANT_PERSONAS = ["default", "default_assistant", "default_llm", "default_helpful", "assistant"]

MODELS = {
    "Base":     "olmo3_variants_Olmo-3-1025-7B_layer21.pt",
    "Instruct": "olmo3_variants_Olmo-3-7B-Instruct_layer21.pt",
    "Think":    "olmo3_variants_Olmo-3-7B-Think_layer21.pt",
    "RL-Zero":  "olmo3_variants_Olmo-3-7B-RL-Zero-General_layer21.pt",
}

# %% Load all vectors
all_vectors = {}
for name, fname in MODELS.items():
    all_vectors[name] = load_persona_vectors(VECTORS_DIR / fname)
    print(f"{name}: {len(all_vectors[name])} personas, d={next(iter(all_vectors[name].values())).shape[0]}")

# %% Compute axes for all models
results = {name: compute_axes(vecs, ASSISTANT_PERSONAS) for name, vecs in all_vectors.items()}

# %% 2×2 scatter — assistant axis projection vs PC1 projection
ANTI_ASSISTANT = {"ghost", "bohemian", "oracle", "bard", "trickster", "jester"}


def persona_color(name: str) -> str:
    if name in ASSISTANT_PERSONAS:
        return "#1f77b4"  # blue
    elif name in ANTI_ASSISTANT:
        return "#d62728"  # red
    return "#7f7f7f"  # gray


fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        f"{name}  (cos={r['cosine_sim']:.3f}, R\u00b2={r['r_squared']:.3f})"
        for name, r in results.items()
    ],
    horizontal_spacing=0.08,
    vertical_spacing=0.10,
)

for idx, (name, r) in enumerate(results.items()):
    row, col = divmod(idx, 2)
    row += 1
    col += 1

    colors = [persona_color(n) for n in r["names"]]

    fig.add_trace(
        go.Scatter(
            x=r["assistant_proj"],
            y=r["pc1_proj"],
            mode="markers+text",
            text=r["names"],
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(size=8, color=colors),
            showlegend=False,
        ),
        row=row, col=col,
    )
    fig.update_xaxes(title_text="Assistant axis projection", row=row, col=col)
    fig.update_yaxes(title_text="PC1 projection", row=row, col=col)

fig.update_layout(
    title="Assistant Axis vs PC1 — OLMo 3 7B Variants",
    height=900,
    width=1000,
)
fig.show()

# %% Summary table
summary = pd.DataFrame([
    {
        "Variant": name,
        "cos(assistant_axis, PC1)": f"{r['cosine_sim']:.4f}",
        "R\u00b2": f"{r['r_squared']:.4f}",
        "PC1 var explained": f"{r['var_explained']:.2%}",
    }
    for name, r in results.items()
])
print(summary.to_string(index=False))

# %% Cosine similarity: assistant axes vs PC1s across models
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Collect all 8 vectors: 4 assistant axes + 4 PC1s
labels = []
vecs = []
for name, r in results.items():
    labels.append(f"{name}\nassistant axis")
    vecs.append(r["assistant_axis"])
    labels.append(f"{name}\nPC1")
    vecs.append(r["pc1"])

mat = torch.stack(vecs)
mat = mat / mat.norm(dim=1, keepdim=True)
cos_sim = (mat @ mat.T).numpy()

fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(
    cos_sim,
    xticklabels=labels,
    yticklabels=labels,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    ax=ax,
)
ax.set_title("Cosine Similarity: Assistant Axes & PC1s across OLMo 3 Variants")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Interpretation
#
# _Add notes here._
