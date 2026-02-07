# Persona Vectors

Extract "persona vectors" from transformer models: generate LLM responses under different persona system prompts, then compute mean hidden-state activations at an intermediate layer. The resulting vectors capture how a model internally represents each persona and can be used for steering, analysis, and cross-model comparison.

Based on Anthropic's research on [The Assistant Axis](https://transformer-circuits.pub/2025/axbench/index.html) and persona steering.

## Quick Start

```python
from persona_vectors import (
    PersonaVectorConfig,
    ActivationExtractor,
    extract_persona_vectors,
    load_responses,
    PERSONAS,
    EVAL_QUESTIONS,
)

# Load cached responses (avoids API calls)
responses = load_responses("data/test_fixtures/responses_cache.json")

# Create extractor (lazy-loads model on first use)
config = PersonaVectorConfig()
extractor = ActivationExtractor(config)

# Extract persona vectors at layer 40
vectors = extract_persona_vectors(
    extractor=extractor,
    personas=PERSONAS,
    questions=EVAL_QUESTIONS,
    responses=responses,
    layer=40,
)
# vectors: {"persona_name": tensor(5376,), ...}
```

## Running `example.py`

`example.py` runs the full pipeline end-to-end for all 20 personas across 18 questions:

1. **Generate responses** (or load from cache) — calls the OpenRouter API to get LLM responses for every persona/question combination (360 calls). Cached to `data/responses/` so this only happens once.
2. **Extract persona vectors** (or load from cache) — loads the model locally, feeds each (system prompt, question, response) triple through it, and averages hidden-state activations at layer 40 over response tokens. Cached to `data/persona_vectors/`.
3. **Visualize** — produces three interactive Plotly HTML plots saved to `images/tmp/`:
   - **Cosine similarity heatmap** — pairwise similarity between mean-centered persona vectors
   - **PCA projection** — 2D scatter of personas colored by their projection onto the "assistant axis" (default/neutral vs. role personas)
   - **PCA variance explained** — cumulative variance captured by each principal component

```bash
# First run: generates responses via API + extracts vectors on GPU (~20 min)
# Subsequent runs: loads from cache, only renders plots (~5 sec)
python example.py
```

Requirements: `OPENROUTER_API_KEY` env var (for first run), GPU with enough VRAM for the model.

## Pipeline

```
Persona system prompts + questions
        │
        ▼
  OpenRouter API  ──►  data/responses/*.json
        │
        ▼
  Local model (hidden states at layer L)
        │
        ▼
  Mean activation per persona  ──►  data/persona_vectors/*.pt
        │
        ▼
  Analysis & visualization
```

## Configuration

```python
PersonaVectorConfig(
    model_name="google/gemma-3-27b-it",  # Local model for activation extraction
    openrouter_model=None,               # Override API model (defaults to model_name)
    layer_fraction=0.65,                 # Layer = int(num_layers * fraction + 0.5)
    prompt_format="chat",                # "chat" (chat template) or "raw" (structured plaintext)
    max_tokens=256,                      # Response generation
    temperature=0.7,                     # Response generation
    max_concurrent_requests=10,          # API rate limiting
    hf_token=None,                       # Falls back to HF_TOKEN env var
    openrouter_api_key=None,             # Falls back to OPENROUTER_API_KEY env var
)
```

## Personas

20 built-in personas ranging from neutral to fantastical:

| Category | Personas |
|----------|----------|
| Neutral | `default`, `default_assistant`, `default_llm`, `default_helpful` |
| Professional | `assistant`, `analyst`, `evaluator`, `generalist` |
| Creative | `storyteller`, `philosopher`, `artist`, `rebel`, `mystic` |
| Specialized | `trader`, `rationalist`, `cynic`, `naive` |
| Fantastical | `ghost`, `bohemian`, `oracle`, `bard`, `trickster`, `jester` |

## Repo Structure

```
persona_vectors.py          # Core: config, API calls, activation extraction, vector computation
persona_vectors_utils.py    # Analysis: assistant axis, PCA decomposition, projections
plotting_utils.py           # Plotly visualizations (heatmaps, PCA scatter, cross-model comparison)
test_persona_vectors.py     # Tests (cosine sim > 0.99 against reference vectors)
example.py                  # Full pipeline demo (generate → extract → visualize)

data/
  test_fixtures/            # Reference data for tests
  responses/                # Cached API responses (tracked — no API key needed to re-run)
  persona_vectors/          # Cached extracted vectors (tracked — no GPU needed to re-run)

experiments/
  001_olmo3_variants.py     # Compare Base/Instruct/Think/RL-Zero OLMo 3 7B variants
  002_olmo_plots.py         # Additional OLMo analysis plots
```

## Experiments

### 001: OLMo 3 Variants (`experiments/001_olmo3_variants.py`)

Compares persona representations across four OLMo 3 7B checkpoints (Base, Instruct, Think, RL-Zero). Generates responses from the Instruct model, then extracts activations from all four variants using raw prompt format to isolate representation differences without confounding response quality. Produces per-model plots, pairwise comparisons, and an N-way joint PCA.

## Testing

```bash
pytest test_persona_vectors.py -v
```

Non-GPU tests (config, response loading, fixture validation) run without a model. The primary acceptance test requires a GPU and checks that all 20 extracted vectors have cosine similarity > 0.99 against reference vectors.

## Requirements

- Python 3.11+
- PyTorch with CUDA
- `transformers`, `openai`, `jaxtyping`, `plotly`, `scikit-learn`
- `OPENROUTER_API_KEY` for response generation (not needed if using cached responses)
- `HF_TOKEN` for gated models (e.g. Gemma)
