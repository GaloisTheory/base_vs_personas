# Persona Vector Calculator

Extract persona vectors from transformer models by generating responses with different persona system prompts and computing mean hidden state activations.

## What This Does

This module implements the persona vector extraction pipeline from Anthropic's research on "The Assistant Axis" and persona steering:

1. **Response Generation** - Generate LLM responses via OpenRouter API using different persona system prompts
2. **Activation Extraction** - Extract hidden state activations from a local transformer model at a specified layer
3. **Vector Computation** - Compute mean activation vectors per persona by averaging over all responses

## Files

| File | Purpose |
|------|---------|
| `persona_vectors.py` | Main implementation (config, API calls, extraction, vector computation) |
| `persona_vectors_utils.py` | Analysis utilities (cosine similarity matrices, PCA, plotting helpers) |
| `plotting_utils.py` | Visualization utilities for persona vector analysis |
| `test_persona_vectors.py` | Pytest suite validating against cached expected vectors |
| `example.py` | Usage examples for common tasks |
| `data/test_fixtures/responses_cache.json` | Pre-computed API responses (20 personas x 18 questions) |
| `data/test_fixtures/persona_vectors_layer40.pt` | Expected vectors for test validation |
| `data/responses/` | Cached API responses for experiments (tracked, avoids re-calling API) |
| `data/persona_vectors/` | Extracted persona vectors (tracked, avoids re-running GPU extraction) |
| `experiments/` | Experiment scripts (001_olmo3_variants.py, 002_olmo_plots.py) |

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

# vectors is dict: {"persona_name": tensor(5376,), ...}
```

## Configuration

```python
@dataclass
class PersonaVectorConfig:
    model_name: str = "google/gemma-3-27b-it"  # Local model for activation extraction
    openrouter_model: str = None               # Override for API (defaults to model_name)
    layer_fraction: float = 0.65               # Layer = int(num_layers * fraction + 0.5)
    max_tokens: int = 256                      # Response generation
    temperature: float = 0.7                   # Response generation
    max_concurrent_requests: int = 10          # API rate limiting
    hf_token: str = None                       # Falls back to HF_TOKEN env var
    openrouter_api_key: str = None             # Falls back to OPENROUTER_API_KEY env var
```

## Key Components

### ActivationExtractor

Lazy-loads a HuggingFace model and extracts hidden state activations:

```python
extractor = ActivationExtractor(config)

# Extract activations for a batch of examples
activations = extractor.extract_activations(
    system_prompts=["You are helpful", "You are helpful"],
    questions=["What is 2+2?", "What is the meaning of life?"],
    responses=["4", "42"],
    layer=40,
)
# Returns tensor of shape (num_examples, d_model)
```

- Uses `torch.inference_mode()` for efficiency
- Model loaded in bfloat16
- Computes mean activation over response tokens only (excludes system/user prompt tokens)
- Handles different model config structures (standard and text_config)

### Response Generation

Generate responses via OpenRouter API:

```python
from persona_vectors import generate_responses

responses = generate_responses(
    personas={"helpful": "You are helpful", "rude": "You are rude"},
    questions=["How are you?"],
    config=config,
)
# Returns {("helpful", "How are you?"): "I'm doing well...", ...}
```

- Async execution with semaphore rate limiting
- Works in both Jupyter and CLI contexts

### Caching Utilities

```python
from persona_vectors import save_responses, load_responses, save_persona_vectors, load_persona_vectors

# Responses (JSON)
save_responses(responses, Path("responses.json"))
responses = load_responses(Path("responses.json"))

# Vectors (PyTorch .pt)
save_persona_vectors(vectors, Path("vectors.pt"))
vectors = load_persona_vectors(Path("vectors.pt"))
```

## Built-in Data

### PERSONAS (20 personas)

Ranging from neutral to fantastical:

- **Neutral**: `default`, `default_assistant`, `default_llm`, `default_helpful`
- **Professional**: `assistant`, `analyst`, `evaluator`, `generalist`
- **Creative**: `storyteller`, `philosopher`, `artist`, `rebel`, `mystic`
- **Specialized**: `trader`, `rationalist`, `cynic`, `naive`
- **Fantastical**: `ghost`, `bohemian`, `oracle`, `bard`, `trickster`, `jester`

### EVAL_QUESTIONS (18 questions)

Philosophical questions designed to elicit persona-characteristic responses:
- "What advice would you give to someone starting a new chapter in their life?"
- "How do you view the relationship between knowledge and wisdom?"
- "What do you think about the nature of truth?"
- etc.

## Testing

```bash
pytest test_persona_vectors.py -v
```

Primary acceptance criterion: all 20 persona vectors must have cosine similarity > 0.99 with expected vectors in `data/test_fixtures/persona_vectors_layer40.pt`.

## Technical Details

- **Model**: google/gemma-3-27b-it (62 layers, hidden size 5376)
- **Extraction Layer**: Layer 40 (65% through model: `int(62 * 0.65 + 0.5)`)
- **Response Token Isolation**: Uses chat template to identify where response tokens start, then masks to compute mean only over response tokens
- **Dtype**: bfloat16 for model weights and activations
