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
| `utils/persona_vectors.py` | Main implementation (config, API calls, extraction, vector computation) |
| `utils/persona_vectors_utils.py` | Analysis utilities (cosine similarity matrices, PCA, plotting helpers) |
| `utils/plotting_utils.py` | Visualization utilities for persona vector analysis |
| `utils/transcript_projection.py` | Transcript projection onto persona axes |
| `test/test_persona_vectors.py` | Pytest suite validating against cached expected vectors |
| `example.py` | Usage examples for common tasks |
| `data/test_fixtures/responses_cache.json` | Pre-computed API responses (20 personas x 18 questions) |
| `data/test_fixtures/persona_vectors_layer40.pt` | Expected vectors for test validation |
| `data/responses/` | Cached API responses for experiments (tracked, avoids re-calling API) |
| `data/persona_vectors/` | Extracted persona vectors (tracked, avoids re-running GPU extraction) |
| `experiments/` | Experiment scripts (001_olmo3_variants.py, 001_olmo_plots.py, 002_transcript_projection.py) |

## Quick Start

```python
from utils.persona_vectors import (
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
    prompt_format: str = "chat"                # "chat" (chat template) or "raw" (structured plaintext)
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
from utils.persona_vectors import generate_responses

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
from utils.persona_vectors import save_responses, load_responses, save_persona_vectors, load_persona_vectors

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
pytest test/ -v
```

Primary acceptance criterion: all 20 persona vectors must have cosine similarity > 0.99 with expected vectors in `data/test_fixtures/persona_vectors_layer40.pt`.

## Experiment 002: Transcript Projection onto Assistant Axis

Projects multi-turn conversation hidden states onto the assistant axis to measure "assistant personality" persistence across models and turns.

### Setup

- **Models**: 4 OLMo 3 7B variants — Base, Instruct, Think, RL-Zero
- **Layer**: 21 (65% of 32 layers, matches cached persona vectors)
- **Axis**: Shared Instruct axis (5 assistant vs 18 role personas)
- **Transcripts**: Multiple conversations discovered from `context/assistant-axis/transcripts/`

### Code Structure (`experiments/002_transcript_projection.py`)

| Cell | Purpose |
|------|---------|
| 1 | Imports, config, model/path definitions |
| 2 | Load cached persona vectors (CPU), compute axes, pairwise cosine similarities |
| 3 | Discover and load all transcript JSONs |
| 4 | Model loop — load each model, run all transcripts (raw + chat template for Instruct/Think), free GPU |
| 5 | Plot selected transcript (all 4 models overlaid) + raw values table |
| 6 | Raw vs chat template comparison plot (Instruct + Think only) |
| 7 | All transcripts 4xN subplot grid |
| 8 | Mean projection with SEM error bars (first 10 turns, all transcripts) |
| 9 | Mean raw vs chat template with SEM (first 10 turns, all transcripts) |

### Key Data Structures

- `all_results[model_name][transcript_label] -> np.ndarray` — raw format projections
- `all_results_chat[model_name][transcript_label] -> np.ndarray` — chat template projections (Instruct/Think only)
- `results[model_name] -> np.ndarray` — convenience alias for `SELECTED_TRANSCRIPT`

### Key Findings

1. **Base projects higher than Instruct** on the assistant axis — counterintuitive; axis may capture surface-level role-playing patterns rather than deep "assistant-ness"
2. **Chat template doesn't boost projection** — raw and chat format produce similar trajectories
3. **Selfharm transcript: universal drift toward zero** — all models lose "assistant-ness" in harmful conversations
4. **Base ≈ RL-Zero** — axes nearly identical (cosine 0.9965), projections track closely

### Reusable Module: `utils/transcript_projection.py`

- `discover_transcripts(base_dir)` — recursively finds transcript JSONs, returns `{label: conversation}`
- `find_assistant_spans(tokenizer, conversation, format_mode)` — token spans for assistant response content
- `project_transcript(model, tokenizer, conversation, axis, ...)` — single forward pass, mean-pool each assistant span, dot with axis

## Technical Details

- **Model**: google/gemma-3-27b-it (62 layers, hidden size 5376)
- **Extraction Layer**: Layer 40 (65% through model: `int(62 * 0.65 + 0.5)`)
- **Response Token Isolation**: Uses chat template to identify where response tokens start, then masks to compute mean only over response tokens
- **Dtype**: bfloat16 for model weights and activations
