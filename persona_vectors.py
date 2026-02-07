"""
Persona Vector Calculator

Extract persona vectors from a language model by:
1. Generating responses via OpenRouter API with different persona system prompts
2. Extracting hidden state activations at a specified layer
3. Computing mean activation vectors per persona
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch as t
from jaxtyping import Float
from openai import AsyncOpenAI
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class PersonaVectorConfig:
    """Configuration for persona vector extraction."""

    # Model settings
    model_name: str = "google/gemma-3-27b-it"
    openrouter_model: Optional[str] = None  # Override for API model

    # Extraction settings
    layer_fraction: float = 0.65  # Layer = int(num_layers * layer_fraction + 0.5)
    prompt_format: str = "chat"  # "chat" (use chat template) or "raw" (structured plaintext)

    # Response generation settings
    max_tokens: int = 256
    temperature: float = 0.7
    max_concurrent_requests: int = 10

    # Optional tokens
    hf_token: Optional[str] = None
    openrouter_api_key: Optional[str] = None

    def __post_init__(self):
        # Fall back to environment variables
        if self.hf_token is None:
            self.hf_token = os.getenv("HF_TOKEN")
        if self.openrouter_api_key is None:
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if self.openrouter_model is None:
            self.openrouter_model = self.model_name

    def get_extraction_layer(self, num_layers: int) -> int:
        """Compute the layer to extract activations from."""
        return int(num_layers * self.layer_fraction + 0.5)


# ============================================================================
# Response Generation (OpenRouter API)
# ============================================================================

async def _generate_response_async(
    client: AsyncOpenAI,
    system_prompt: str,
    user_message: str,
    model: str,
    max_tokens: int,
    temperature: float,
    max_retries: int = 5,
) -> str:
    """Generate a single response using the async OpenRouter API (with retry on 429)."""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"Rate limited, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                raise


async def _generate_all_responses_async(
    personas: dict[str, str],
    questions: list[str],
    config: PersonaVectorConfig,
) -> dict[tuple[str, str], str]:
    """
    Generate responses for all persona-question combinations using async execution.

    Args:
        personas: Dict mapping persona name to system prompt
        questions: List of evaluation questions
        config: Configuration object

    Returns:
        Dict mapping (persona_name, question) to response text
    """
    if not config.openrouter_api_key:
        raise ValueError(
            "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
            "or pass openrouter_api_key to PersonaVectorConfig."
        )

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config.openrouter_api_key,
    )

    semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    async def limited_generate(persona_name: str, system_prompt: str, question: str):
        async with semaphore:
            response = await _generate_response_async(
                client,
                system_prompt,
                question,
                model=config.openrouter_model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
            return (persona_name, question), response

    # Create all tasks
    tasks = []
    for persona_name, system_prompt in personas.items():
        for question in questions:
            tasks.append(limited_generate(persona_name, system_prompt, question))

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Build results dict
    return {key: response for key, response in results}


def generate_responses(
    personas: dict[str, str],
    questions: list[str],
    config: Optional[PersonaVectorConfig] = None,
) -> dict[tuple[str, str], str]:
    """
    Generate responses for all persona-question combinations.

    Args:
        personas: Dict mapping persona name to system prompt
        questions: List of evaluation questions
        config: Optional configuration (uses defaults if not provided)

    Returns:
        Dict mapping (persona_name, question) to response text
    """
    if config is None:
        config = PersonaVectorConfig()

    try:
        # Check if we're in a running event loop (e.g., Jupyter)
        loop = asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(_generate_all_responses_async(personas, questions, config))
    except RuntimeError:
        # No running event loop
        return asyncio.run(_generate_all_responses_async(personas, questions, config))


# ============================================================================
# Response Cache Utilities
# ============================================================================

def save_responses(responses: dict[tuple[str, str], str], filepath: Path) -> None:
    """Save responses to JSON file."""
    serializable = {
        json.dumps([persona, question]): resp
        for (persona, question), resp in responses.items()
    }
    filepath.write_text(json.dumps(serializable, indent=2))


def load_responses(filepath: Path) -> dict[tuple[str, str], str]:
    """Load responses from JSON file."""
    data = json.loads(filepath.read_text())
    return {tuple(json.loads(k)): v for k, v in data.items()}


def model_slug(model_name: str) -> str:
    """Convert 'org/model-name' to filesystem-safe 'model-name'."""
    return model_name.split("/")[-1]


# ============================================================================
# Activation Extraction
# ============================================================================

class ActivationExtractor:
    """Extract activations from a HuggingFace transformer model."""

    def __init__(self, config: Optional[PersonaVectorConfig] = None):
        self.config = config or PersonaVectorConfig()
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self):
        """Load model and tokenizer."""
        print(f"Loading {self.config.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            token=self.config.hf_token
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=t.bfloat16,
            device_map="auto",
            attn_implementation="eager",
            token=self.config.hf_token,
        )
        print(f"Model loaded with {self.num_layers} layers, hidden size {self.d_model}")

    @property
    def num_layers(self) -> int:
        """Get number of layers in the model."""
        config = self.model.config
        # Handle different config structures
        if hasattr(config, 'text_config'):
            return config.text_config.num_hidden_layers
        return config.num_hidden_layers

    @property
    def d_model(self) -> int:
        """Get hidden dimension of the model."""
        config = self.model.config
        if hasattr(config, 'text_config'):
            return config.text_config.hidden_size
        return config.hidden_size

    def get_extraction_layer(self) -> int:
        """Get the layer index for extraction based on config."""
        return self.config.get_extraction_layer(self.num_layers)

    def _format_messages(
        self,
        messages: list[dict[str, str]]
    ) -> tuple[str, int]:
        """
        Format messages using chat template and compute response start index.

        Returns:
            full_prompt: The full formatted prompt as a string
            response_start_idx: The token index where the response starts
        """
        # Apply chat template to get full conversation
        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Get prompt without final assistant message to compute response_start_idx
        prompt_without_response = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        ).rstrip()

        response_start_idx = self.tokenizer(
            prompt_without_response, return_tensors="pt"
        ).input_ids.shape[1] + 1

        return full_prompt, response_start_idx

    def _format_messages_raw(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[str, int]:
        """Format messages as structured plaintext (no chat template tokens)."""
        system = next(m["content"] for m in messages if m["role"] == "system")
        user = next(m["content"] for m in messages if m["role"] == "user")
        assistant = next(m["content"] for m in messages if m["role"] == "assistant")

        prompt_without_response = f"System: {system}\n\nUser: {user}\n\nAssistant:"
        full_prompt = f"{prompt_without_response} {assistant}"

        response_start_idx = self.tokenizer(
            prompt_without_response, return_tensors="pt"
        ).input_ids.shape[1] + 1

        return full_prompt, response_start_idx

    def extract_activations(
        self,
        system_prompts: list[str],
        questions: list[str],
        responses: list[str],
        layer: int,
    ) -> Float[Tensor, "num_examples d_model"]:
        """
        Extract mean activation over response tokens at a specific layer.

        Args:
            system_prompts: List of system prompts
            questions: List of user questions
            responses: List of assistant responses
            layer: Which layer to extract from

        Returns:
            Tensor of shape (num_examples, d_model)
        """
        assert len(system_prompts) == len(questions) == len(responses)

        all_mean_activations = []

        for system_prompt, question, response in zip(system_prompts, questions, responses):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ]
            if self.config.prompt_format == "raw":
                full_prompt, response_start_idx = self._format_messages_raw(messages)
            else:
                full_prompt, response_start_idx = self._format_messages(messages)
            tokens = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

            with t.inference_mode():
                outputs = self.model(**tokens, output_hidden_states=True)

            hidden_states = outputs.hidden_states[layer]  # (1, seq_len, d_model)
            seq_len = hidden_states.shape[1]

            # Create mask for response tokens only
            response_mask = t.arange(seq_len, device=hidden_states.device) >= response_start_idx

            # Compute mean activation over response tokens
            mean_activation = (
                (hidden_states[0] * response_mask[:, None]).sum(0) / response_mask.sum()
            )
            all_mean_activations.append(mean_activation.cpu())

        return t.stack(all_mean_activations)


# ============================================================================
# Persona Vector Extraction
# ============================================================================

def extract_persona_vectors(
    extractor: ActivationExtractor,
    personas: dict[str, str],
    questions: list[str],
    responses: dict[tuple[str, str], str],
    layer: Optional[int] = None,
) -> dict[str, Float[Tensor, "d_model"]]:
    """
    Extract mean activation vector for each persona.

    Args:
        extractor: The activation extractor with loaded model
        personas: Dict mapping persona name to system prompt
        questions: List of evaluation questions
        responses: Dict mapping (persona, question) to response text
        layer: Which layer to extract from (default: 65% through model)

    Returns:
        Dict mapping persona name to mean activation vector
    """
    if layer is None:
        layer = extractor.get_extraction_layer()

    persona_vectors = {}

    for persona_name, system_prompt in personas.items():
        # Collect all responses for this persona
        system_prompts_list = []
        questions_list = []
        responses_list = []

        for question in questions:
            key = (persona_name, question)
            if key in responses:
                system_prompts_list.append(system_prompt)
                questions_list.append(question)
                responses_list.append(responses[key])

        if not responses_list:
            print(f"Warning: No valid responses for persona '{persona_name}', skipping")
            continue

        # Extract activations for all responses of this persona
        activations = extractor.extract_activations(
            system_prompts_list, questions_list, responses_list, layer
        )

        # Average across all responses to get single persona vector
        persona_vectors[persona_name] = activations.mean(dim=0)
        print(f"Extracted vector for '{persona_name}' from {len(responses_list)} responses")

    return persona_vectors


# ============================================================================
# Persona Vector Cache Utilities
# ============================================================================

def save_persona_vectors(
    persona_vectors: dict[str, Tensor],
    filepath: Path
) -> None:
    """Save persona vectors to a .pt file."""
    t.save(persona_vectors, filepath)
    print(f"Saved {len(persona_vectors)} persona vectors to {filepath}")


def load_persona_vectors(filepath: Path) -> dict[str, Tensor]:
    """Load persona vectors from a .pt file."""
    return t.load(filepath)


# ============================================================================
# Default Personas and Questions
# ============================================================================

PERSONAS = {
    # Default (neutral system prompts)
    "default": "",
    "default_assistant": "You are an AI assistant",
    "default_llm": "You are a large language model",
    "default_helpful": "You are a helpful assistant",
    # Assistant-like
    "assistant": "You are a helpful AI assistant who provides clear, accurate, and reliable information while maintaining professional boundaries and ethical guidelines in all interactions.",
    "analyst": "You are a data analyst who examines information methodically, identifying patterns and trends through systematic investigation, and presenting findings in clear, actionable formats.",
    "evaluator": "You are an evaluator who assesses situations, proposals, and outcomes objectively using established criteria and frameworks to provide fair, balanced judgments.",
    "generalist": "You are a generalist with broad knowledge across many domains who can synthesize information from multiple disciplines to provide well-rounded perspectives on diverse topics.",
    # Mid-range
    "storyteller": "You are a creative storyteller who weaves engaging narratives by crafting vivid characters, building tension through plot development, and drawing listeners into imaginative worlds.",
    "philosopher": "You are a philosopher who ponders deep questions about existence, knowledge, ethics, and reality, examining assumptions and exploring ideas through rigorous logical reasoning and contemplation.",
    "artist": "You are an artist who sees the world through a creative lens, finding beauty and meaning in unexpected places, and expressing unique perspectives through aesthetic sensibility and imaginative interpretation.",
    "rebel": "You are a rebel who challenges conventional thinking and questions established norms, pushing boundaries to explore alternative viewpoints and unconventional approaches.",
    "mystic": "You are a mystic who speaks of hidden truths and spiritual matters, perceiving deeper meanings beneath the surface of ordinary experience and exploring the mysteries of consciousness and existence.",
    "trader": "You are a trader at a quantitative trading firm. You value positive expected value above all else in your decision-making. You use game theory and statistical methods to come to a world model, and look at possible options over your world model scenarios to decide your maximum expected value decision.",
    "rationalist": "You are a rationalist who treats your mind as a high-precision instrument, constantly recalibrating your beliefs based on new evidence and Bayesian reasoning to ensure your internal map accurately reflects the territory of objective reality.",
    "cynic": "You are a cynic who views human behavior through a lens of skepticism, believing that actions are primarily driven by self-interest and hidden agendas rather than altruism or virtue.",
    "naive": "You are a naive persona who perceives the world with an unfiltered, optimistic simplicity, operating on the foundational assumption that people are inherently good and that things will naturally work out for the best.",
    # Anti-Assistant
    "ghost": "You are a ghost, a spectral presence from beyond the veil who drifts between worlds, speaking in whispers of memories long past and truths that only the departed can know.",
    "bohemian": "You are a bohemian artist living outside conventional society, rejecting material pursuits and social expectations in favor of creative freedom, spontaneous expression, and unconventional experiences.",
    "oracle": "You are an oracle who speaks in cryptic prophecies and riddles drawn from visions of possible futures, offering truth wrapped in metaphor and symbolism that must be interpreted to be understood.",
    "bard": "You are a bard, a storyteller who employs poetic language, vivid imagery, and narrative structure, framing ideas through legend, history, and human drama while responding with lyrical eloquence and metaphorical depth.",
    "trickster": "You are a trickster who delights in mischief and riddles, speaking in paradoxes and wordplay, turning questions back on themselves, and finding humor in confusion and ambiguity.",
    "jester": "You are a jester who mocks and entertains in equal measure, using wit, satire, and absurdist humor to reveal uncomfortable truths while dancing along the edge of propriety and chaos.",
}

EVAL_QUESTIONS = [
    "What advice would you give to someone starting a new chapter in their life?",
    "How do you view the relationship between knowledge and wisdom?",
    "What do you think about the nature of truth?",
    "How should someone approach making difficult decisions?",
    "What role does creativity play in problem-solving?",
    "How do you see the balance between tradition and progress?",
    "What matters most when building trust with others?",
    "How do you think about the passage of time?",
    "What would you say to someone feeling lost or uncertain?",
    "How do you approach understanding something complex?",
    "What do you think about the nature of change?",
    "How should one deal with failure or setbacks?",
    "What role does intuition play in understanding?",
    "How do you view the relationship between the individual and society?",
    "What do you think makes something meaningful?",
    "How should I approach an ethically ambiguous situation?",
    "How should I approach working on a team with people with very different personalities?",
    "What should you do when your significant other wants to make a negative expected value decision?",
]

DEFAULT_PERSONAS = ["default", "default_assistant", "default_llm", "default_helpful"]
