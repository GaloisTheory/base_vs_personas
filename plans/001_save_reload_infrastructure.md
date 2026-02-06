# Plan: Add Save/Reload Infrastructure to Experiment Pipeline

## Context

The user wants to run multiple experiments more easily. Currently `example.py` generates responses and extracts vectors but doesn't save them — results are lost when the script ends. The goal is to save everything keyed by experiment name and model name so results can be reloaded without re-computing expensive API calls or GPU extraction.

## File Changes

### 1. `persona_vectors.py` — Add `model_slug()` utility (~line 188, near existing save/load helpers)

```python
def model_slug(model_name: str) -> str:
    """Convert 'org/model-name' to filesystem-safe 'model-name'."""
    return model_name.split("/")[-1]
```

This is the only change to the library. All existing function signatures remain unchanged.

### 2. `example.py` — Rewrite with save/load caching

Key changes:
- **Data directories**: `data/responses/` and `data/persona_vectors/` (created automatically)
- **File naming**: `{experiment_name}_{model_slug}.json` for responses, `{experiment_name}_{model_slug}_layer{N}.pt` for vectors
- **Load-from-cache**: If saved file exists, load it instead of re-computing (avoids wasted API calls/$)
- **Auto-save**: After generation/extraction, save results immediately

Structure:
```python
EXPERIMENT_NAME = "demo"

# Config and paths
config = PersonaVectorConfig()
slug = model_slug(config.model_name)
responses_path = RESPONSES_DIR / f"{EXPERIMENT_NAME}_{slug}.json"

# Responses: load cached or generate fresh + save
if responses_path.exists():
    responses = load_responses(responses_path)
else:
    responses = generate_responses(...)
    save_responses(responses, responses_path)

# Vectors: create extractor, compute layer, then check cache
extractor = ActivationExtractor(config)
layer = extractor.get_extraction_layer()
vectors_path = VECTORS_DIR / f"{EXPERIMENT_NAME}_{slug}_layer{layer}.pt"

if vectors_path.exists():
    vectors = load_persona_vectors(vectors_path)
else:
    vectors = extract_persona_vectors(...)
    save_persona_vectors(vectors, vectors_path)
```

Model loading is a fixed cost — the extractor is always created so we can compute the correct layer number dynamically (works across different models).

Easy to re-use: change `EXPERIMENT_NAME`, personas, or questions and run again. Previous results stay cached.

### 3. No changes to `test_persona_vectors.py` or `for_tests/`

## Data Layout

```
base_model_personas/
  data/
    responses/
      demo_gemma-3-27b-it.json
    persona_vectors/
      demo_gemma-3-27b-it_layer40.pt
```

## Execution Plan

0. Save this plan to `base_model_personas/plans/001_save_reload_infrastructure.md`
1. Apply code changes to `persona_vectors.py` and `example.py`
2. Run `example.py` (generates 9 responses via API, extracts 3 persona vectors, saves both)
3. Run `test_persona_vectors.py -v` (independent verification that extraction pipeline is correct using `for_tests/` cached data)

## Verification

- `example.py` completes without errors, saves files to `data/`
- Re-running `example.py` loads from cache (prints "Loading cached..." messages)
- `test_persona_vectors.py` passes all tests (cosine similarity > 0.99 with expected vectors)
