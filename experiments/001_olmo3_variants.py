# Experiment 002: Persona Vectors Across OLMo 3 7B Variants
#
# Compare how Base, Instruct, Think, and RL-Zero variants represent personas.
# Design: generate responses from Instruct via OpenRouter, extract activations
# from all four variants using the same responses + raw prompt format.
# This isolates representation differences without confounding response quality.
#
# See: plns/002_olmo3_variants.md
# %%
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_vectors import (
    PersonaVectorConfig,
    ActivationExtractor,
    generate_responses,
    extract_persona_vectors,
    save_responses,
    load_responses,
    save_persona_vectors,
    load_persona_vectors,
    model_slug,
    PERSONAS,
    EVAL_QUESTIONS,
)
from plotting_utils import (
    plot_cosine_similarity_heatmap,
    plot_pca_projection,
    plot_pca_variance_explained,
    plot_pca_comparison,
    plot_cosine_cross_similarity,
    plot_pca_comparison_nway,
)

# ===== EXPERIMENT CONFIG =====
EXPERIMENT_NAME = "olmo3_variants"

PILOT_MODE = False

# Pilot subsampling: 8 personas × 6 questions
PILOT_PERSONAS = {k: PERSONAS[k] for k in [
    "default", "default_helpful",   # neutral baseline
    "assistant", "analyst",          # professional
    "philosopher", "rebel",          # mid-range
    "oracle", "jester",              # fantastical
]}
PILOT_QUESTIONS = EVAL_QUESTIONS[:6]

personas = PILOT_PERSONAS if PILOT_MODE else PERSONAS
questions = PILOT_QUESTIONS if PILOT_MODE else EVAL_QUESTIONS
pilot_tag = "_pilot" if PILOT_MODE else ""

# Response generation (Instruct model via OpenRouter)
RESPONSE_CONFIG = PersonaVectorConfig(
    model_name="allenai/Olmo-3-7B-Instruct",
    openrouter_model="allenai/olmo-3-7b-instruct",
    max_concurrent_requests=3,
)

# Four model variants — all use raw prompt format for controlled comparison
MODEL_CONFIGS = {
    "Base": PersonaVectorConfig(
        model_name="allenai/Olmo-3-1025-7B",
        prompt_format="raw",
    ),
    "Instruct": PersonaVectorConfig(
        model_name="allenai/Olmo-3-7B-Instruct",
        prompt_format="raw",
    ),
    "Think": PersonaVectorConfig(
        model_name="allenai/Olmo-3-7B-Think",
        prompt_format="raw",
    ),
    "RL-Zero": PersonaVectorConfig(
        model_name="allenai/Olmo-3-7B-RL-Zero-General",
        prompt_format="raw",
    ),
}
# ===== END CONFIG =====

PROJECT_DIR = Path(__file__).parent.parent
RESPONSES_DIR = PROJECT_DIR / "data" / "responses"
VECTORS_DIR = PROJECT_DIR / "data" / "persona_vectors"
OUTPUT_DIR = PROJECT_DIR / "images" / f"{EXPERIMENT_NAME}{pilot_tag}"

RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
VECTORS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% --- 1. Generate / load responses (Instruct model via OpenRouter) ---
response_slug = model_slug(RESPONSE_CONFIG.model_name)
responses_path = RESPONSES_DIR / f"{EXPERIMENT_NAME}_{response_slug}{pilot_tag}.json"

if responses_path.exists():
    print(f"Loading cached responses from {responses_path}")
    responses = load_responses(responses_path)
    print(f"Loaded {len(responses)} cached responses")
else:
    print("Generating responses via OpenRouter API...")
    responses = generate_responses(personas, questions, RESPONSE_CONFIG)
    save_responses(responses, responses_path)
    print(f"Generated and saved {len(responses)} responses to {responses_path}")

# %% --- 2. Extract vectors from all 4 model variants (sequential) ---
all_vectors: dict[str, dict[str, torch.Tensor]] = {}

OLMO3_NUM_LAYERS = 32  # All OLMo3 7B variants share this architecture

for variant_name, config in MODEL_CONFIGS.items():
    slug = model_slug(config.model_name)
    layer = config.get_extraction_layer(OLMO3_NUM_LAYERS)
    vectors_path = VECTORS_DIR / f"{EXPERIMENT_NAME}_{slug}_layer{layer}{pilot_tag}.pt"

    if vectors_path.exists():
        print(f"Loading cached {variant_name} vectors from {vectors_path}")
        all_vectors[variant_name] = load_persona_vectors(vectors_path)
        print(f"Loaded {len(all_vectors[variant_name])} cached persona vectors")
    else:
        print(f"Extracting {variant_name} vectors at layer {layer}...")
        extractor = ActivationExtractor(config)
        all_vectors[variant_name] = extract_persona_vectors(
            extractor=extractor,
            personas=personas,
            questions=questions,
            responses=responses,
            layer=layer,
        )
        save_persona_vectors(all_vectors[variant_name], vectors_path)

        # Free GPU memory before loading next model
        print(f"Freeing {variant_name} model from GPU...")
        del extractor
        torch.cuda.empty_cache()

# %% --- 3. Per-model plots (3 per model = 12 total) ---
print("Generating per-model plots...")

for variant_name, vectors in all_vectors.items():
    tag = variant_name.lower().replace("-", "_")
    plot_cosine_similarity_heatmap(
        vectors, save_path=OUTPUT_DIR / f"cosine_sim_{EXPERIMENT_NAME}_{tag}{pilot_tag}.html"
    )
    plot_pca_projection(
        vectors, save_path=OUTPUT_DIR / f"pca_{EXPERIMENT_NAME}_{tag}{pilot_tag}.html"
    )
    plot_pca_variance_explained(
        vectors, save_path=OUTPUT_DIR / f"pca_var_{EXPERIMENT_NAME}_{tag}{pilot_tag}.html"
    )

# %% --- 4. Pairwise comparison plots (6 pairs × 2 plot types = 12 total) ---
print("Generating pairwise comparison plots...")

variant_names = list(all_vectors.keys())
for name_a, name_b in itertools.combinations(variant_names, 2):
    tag_a = name_a.lower().replace("-", "_")
    tag_b = name_b.lower().replace("-", "_")
    pair_tag = f"{tag_a}_vs_{tag_b}"

    plot_pca_comparison(
        all_vectors[name_a], all_vectors[name_b],
        label_a=name_a, label_b=name_b,
        save_path=OUTPUT_DIR / f"pca_comparison_{EXPERIMENT_NAME}_{pair_tag}{pilot_tag}.html",
    )
    plot_cosine_cross_similarity(
        all_vectors[name_a], all_vectors[name_b],
        label_a=name_a, label_b=name_b,
        save_path=OUTPUT_DIR / f"cross_similarity_{EXPERIMENT_NAME}_{pair_tag}{pilot_tag}.html",
    )

# %% --- 5. N-way joint PCA (all 4 models) ---
print("Generating N-way joint PCA plot...")

plot_pca_comparison_nway(
    all_vectors=list(all_vectors.values()),
    labels=list(all_vectors.keys()),
    save_path=OUTPUT_DIR / f"pca_nway_{EXPERIMENT_NAME}{pilot_tag}.html",
)

print(f"All plots saved to {OUTPUT_DIR}")
