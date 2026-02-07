# Persona Vectors - Full Pipeline
# Runs all 20 personas x 18 questions through: API generation -> vector extraction -> visualization
# Results are cached to data/ so expensive steps (API calls, GPU extraction) aren't repeated.
# %%
# Path fix for running from any directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.persona_vectors import (
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
from utils.plotting_utils import (
    plot_cosine_similarity_heatmap,
    plot_pca_projection,
    plot_pca_variance_explained,
)

# %%  --- Settings ---
EXPERIMENT_NAME = "olmo3_variants"
PROJECT_DIR = Path(__file__).parent
RESPONSES_DIR = PROJECT_DIR / "data" / "responses"
VECTORS_DIR = PROJECT_DIR / "data" / "persona_vectors"
OUTPUT_DIR = PROJECT_DIR / "images" / "tmp"

RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
VECTORS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Full config: 20 personas x 18 questions = 360 API calls
config = PersonaVectorConfig()
config.model_name = "allenai/Olmo-3-7B-Instruct"
slug = model_slug(config.model_name)
responses_path = RESPONSES_DIR / f"{EXPERIMENT_NAME}_{slug}.json"
# %%
# --- Responses: load cached or generate fresh + save ---
if responses_path.exists():
    print(f"Loading cached responses from {responses_path}")
    responses = load_responses(responses_path)
    print(f"Loaded {len(responses)} cached responses")
else:
    print("Generating responses via OpenRouter API...")
    responses = generate_responses(PERSONAS, EVAL_QUESTIONS, config)
    save_responses(responses, responses_path)
    print(f"Generated and saved {len(responses)} responses to {responses_path}")

# %%
# --- Vectors: create extractor, compute layer, then check cache ---
extractor = ActivationExtractor(config)
layer = extractor.get_extraction_layer()
vectors_path = VECTORS_DIR / f"{EXPERIMENT_NAME}_{slug}_layer{layer}.pt"

if vectors_path.exists():
    print(f"Loading cached persona vectors from {vectors_path}")
    vectors = load_persona_vectors(vectors_path)
    print(f"Loaded {len(vectors)} cached persona vectors")
else:
    print(f"Extracting persona vectors at layer {layer}...")
    vectors = extract_persona_vectors(
        extractor=extractor,
        personas=PERSONAS,
        questions=EVAL_QUESTIONS,
        responses=responses,
        layer=layer,
    )
    save_persona_vectors(vectors, vectors_path)
    print(f"Extracted and saved {len(vectors)} persona vectors to {vectors_path}")

# %%  --- Visualize ---
plot_cosine_similarity_heatmap(vectors, save_path=OUTPUT_DIR / f"cosine_similarity_{EXPERIMENT_NAME}.html").show()
plot_pca_projection(vectors, save_path=OUTPUT_DIR / f"pca_projection_{EXPERIMENT_NAME}.html").show()
plot_pca_variance_explained(vectors, save_path=OUTPUT_DIR / f"pca_variance_{EXPERIMENT_NAME}.html").show()
print(f"Plots saved to {OUTPUT_DIR}")
