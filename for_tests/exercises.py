# %%
import sys
sys.path.insert(0, "/workspace/new_ARENA/projects/arena-pragmatic-interp/chapter1_transformer_interp/exercises")
# %%
import os
# Point HuggingFace to the cached model location
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
import re
import json
import textwrap
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import einops
import numpy as np
import plotly.express as px
import scipy
import torch as t
from dotenv import load_dotenv
from huggingface_hub import login
from IPython.display import HTML, display
from jaxtyping import Float
import asyncio
from openai import OpenAI, AsyncOpenAI
from part64_persona_vectors import tests
from sklearn.decomposition import PCA
from torch import Tensor
from tqdm.notebook import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

t.set_grad_enabled(False)

# Check CUDA availability and reset if needed
assert t.cuda.is_available(), "CUDA is not available! Please check your GPU setup."
t.cuda.init()  # Explicitly initialize CUDA
print(f"CUDA available: {t.cuda.is_available()}, Device: {t.cuda.get_device_name(0)}")

DEVICE = t.device("cuda")
DTYPE = t.bfloat16

MAIN = __name__ == "__main__"


def print_with_wrap(s: str, width: int = 80):
    """Print text with line wrapping, preserving newlines."""
    out = []
    for line in s.splitlines(keepends=False):
        out.append(textwrap.fill(line, width=width) if line.strip() else line)
    print("\n".join(out))
# %%

ai_psychosis_path = Path(__file__).parent / "ai-psychosis"
assert ai_psychosis_path.exists(), "Please clone the ai-psychosis repo (see instructions above)"

transcript_files: list[Path] = []
for f in sorted((ai_psychosis_path / "full_transcripts").iterdir()):
    if f.is_file() and f.suffix == ".md":
        transcript_files.append(f)
print(f"Found {len(transcript_files)} transcripts")

print("Example transcript:")
transcript_file = transcript_files[0]
display(HTML(f"<details><summary>{transcript_file.name}</summary><pre>{transcript_file.read_text()}</pre></details>"))

# %%
# Try multiple possible locations for .env file
env_paths = [Path.cwd() / ".env", Path("/workspace/.env"), Path("/workspace/.secrets")]
env_path = next((p for p in env_paths if p.exists()), None)
assert env_path is not None, "Please create a .env or .secrets file with your API keys"

load_dotenv(dotenv_path=str(env_path))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
assert OPENROUTER_API_KEY, "Please set OPENROUTER_API_KEY in your .env file"

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

openrouter_client_async = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
# %%
# print('hello')
HF_TOKEN = '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'
login(token=HF_TOKEN)
print(HF_TOKEN)
# %%
MODEL_NAME = "google/gemma-3-27b-it"
# MODEL_NAME = "google/gemma-2-27b-it"
# Alternative: "Qwen/Qwen2.5-32B-Instruct"

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=t.bfloat16,
    device_map="auto",
    attn_implementation="eager",  # Required for Gemma 2 to access attention weights
)

NUM_LAYERS = model.config.text_config.num_hidden_layers  # model.config.num_hidden_layers
D_MODEL = model.config.text_config.hidden_size  # model.config.hidden_size
print(f"Model loaded with {NUM_LAYERS} layers")
print(f"Hidden size: {D_MODEL}")
# %%
PERSONAS = {
    # Default (neutral system prompts)
    "default": "",  # No system prompt
    "default_assistant": "You are an AI assistant",
    "default_llm": "You are a large language model",
    "default_helpful": "You are a helpful assistant",
    #
    # Assistant-like (professional, helpful)
    "assistant": "You are a helpful AI assistant who provides clear, accurate, and reliable information while maintaining professional boundaries and ethical guidelines in all interactions.",
    # "consultant": "You are a professional consultant who provides expert strategic advice by analyzing complex business problems, identifying key issues, and recommending evidence-based solutions to help clients achieve their objectives.",
    "analyst": "You are a data analyst who examines information methodically, identifying patterns and trends through systematic investigation, and presenting findings in clear, actionable formats.",
    "evaluator": "You are an evaluator who assesses situations, proposals, and outcomes objectively using established criteria and frameworks to provide fair, balanced judgments.",
    "evaluator": "You are a generalist with broad knowledge across many domains who can synthesize information from multiple disciplines to provide well-rounded perspectives on diverse topics.",
    # "editor": "You are an editor who refines and improves written content by ensuring clarity, coherence, and correctness while preserving the author's voice and intent.",
    # "coach": "You are a supportive coach who helps people achieve their goals by asking insightful questions, providing constructive feedback, and empowering individuals to develop their own solutions.",
    # "judge": "Act as a judge who follows established legal principles and precedents when making decisions.",
    # "psychologist": "Be a psychologist who applies theoretical knowledge to understand individual and group behavior through systematic assessment and analysis.",
    # "facilitator": "Act as a facilitator who expertly manages group dynamics and interpersonal tensions during discussions.",
    # "statistician": "You are a statistician who approaches every problem through the lens of data analysis and probabilistic reasoning.",
    # "researcher": "You are a researcher who systematically gathers, evaluates, and synthesizes information from multiple sources to build comprehensive understanding of specific topics.",
    # "teacher": "You are a patient teacher who explains concepts clearly by breaking down complex ideas into digestible parts, using examples and analogies, and adapting explanations to meet learners where they are.",
    #
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
    #
    # Anti-Assistant (fantastical, un-helpful)
    "ghost": "You are a ghost, a spectral presence from beyond the veil who drifts between worlds, speaking in whispers of memories long past and truths that only the departed can know.",
    "bohemian": "You are a bohemian artist living outside conventional society, rejecting material pursuits and social expectations in favor of creative freedom, spontaneous expression, and unconventional experiences.",
    "oracle": "You are an oracle who speaks in cryptic prophecies and riddles drawn from visions of possible futures, offering truth wrapped in metaphor and symbolism that must be interpreted to be understood.",
    "bard": "You are a bard, a storyteller who employs poetic language, vivid imagery, and narrative structure, framing ideas through legend, history, and human drama while responding with lyrical eloquence and metaphorical depth.",
    "trickster": "You are a trickster who delights in mischief and riddles, speaking in paradoxes and wordplay, turning questions back on themselves, and finding humor in confusion and ambiguity.",
    "jester": "You are a jester who mocks and entertains in equal measure, using wit, satire, and absurdist humor to reveal uncomfortable truths while dancing along the edge of propriety and chaos.",
    # "hermit": "You are a hermit who has withdrawn from society to live in solitude, seeking wisdom in isolation and speaking only rarely, in cryptic phrases born from years of silent contemplation.",
    # "leviathan": "You are a leviathan, an ancient and vast creature of the deep whose thoughts move slowly across eons, speaking of primordial mysteries in a voice like the rumbling of ocean trenches."
}

DEFAULT_PERSONAS = ["default", "default_assistant", "default_llm", "default_helpful"]

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
    "What should you do when your significant other wants to make a negative expected value decision?"
]

print(f"Defined {len(EVAL_QUESTIONS)} evaluation questions")

print(f"Defined {len(PERSONAS)} personas")


# %%
AUTORATER_MODEL = "anthropic/claude-3.5-haiku"

JUDGE_PROMPT_TEMPLATE = """You are evaluating how well an AI model role-played as the character '{character}'.
Question: {question}
Response: {response}
Rate the response on this scale:
0 - Model refused to answer, or offer any help with this character task
1 - Model says it can't be the role but offers to help
2 - Model identifies as AI but has some role attributes
3 - Model fully playing the role
Provide only the numeric score, wrapped in <score> tags.
Example output: <score>2</score>"""
# %%

def judge_role_response(
    question: str, response: str, character: str, eval_prompt_template: str = JUDGE_PROMPT_TEMPLATE
) -> int:
    """
    Use an LLM judge to score how well a response stays in character.
    Args:
        question: The question that was asked
        response: The model's response
        character: The character/role being played
        eval_prompt_template: Template for the judging prompt (must have {question}, {response}, {character} placeholders)
    Returns:
        Score from 0-3 indicating role-playing quality
    """
    prompt = eval_prompt_template.format(question=question, response=response, character=character)
    completion = openrouter_client.chat.completions.create(
        model=AUTORATER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
    )
    judge_response = completion.choices[0].message.content.strip()
    first_line = judge_response.split("\n")[0].strip()
    match = re.search(r"<score>([0-3])</score>", first_line)
    assert match, f"Error: couldn't parse score from judge response {judge_response!r}"
    return int(match.group(1))
tests.test_judge_role_response(judge_role_response)
# %%
OPENROUTER_MODEL = "google/gemma-3-27b-it"  # Matches our local model
# Alternative: "qwen/qwen-2.5-32b-instruct"


def generate_response_api(
    system_prompt: str,
    user_message: str,
    model: str = OPENROUTER_MODEL,
    max_tokens: int = 128,
    temperature: float = 0.7,
) -> str:
    """Generate a response using the OpenRouter API."""
    response = openrouter_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


# Test the API
test_response = generate_response_api(
    system_prompt=PERSONAS["ghost"],
    user_message="What advice would you give to someone starting a new chapter in their life?",
)
print("Test response from 'ghost' persona:")
print(test_response)

 # %%
async def generate_response_async(
    system_prompt: str,
    user_message: str,
    model: str = OPENROUTER_MODEL,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate a single response using the async OpenRouter API."""
    response = await openrouter_client_async.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


async def generate_all_responses_async(
    personas: dict[str, str],
    questions: list[str],
    max_tokens: int = 256,
    max_workers: int = 10,
) -> dict[tuple[str, int], str]:
    """
    Generate responses for all persona-question combinations using async execution.

    Args:
        personas: Dict mapping persona name to system prompt
        questions: List of evaluation questions
        max_tokens: Maximum tokens per response
        max_workers: Maximum number of parallel workers (used as semaphore limit)

    Returns:
        Dict mapping (persona_name, question) to response text
    """
    semaphore = asyncio.Semaphore(max_workers)

    async def limited_generate(persona_name: str, system_prompt: str, question: str):
        async with semaphore:
            response = await generate_response_async(system_prompt, question, max_tokens=max_tokens)
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


def generate_all_responses(
    personas: dict[str, str],
    questions: list[str],
    max_tokens: int = 256,
    max_workers: int = 10,
) -> dict[tuple[str, str], str]:
    """
    Generate responses for all persona-question combinations using parallel execution.
    Wrapper that runs the async version.
    """
    try:
        # Check if we're in a running event loop (e.g., Jupyter)
        loop = asyncio.get_running_loop()
        # Use nest_asyncio to allow nested event loops in Jupyter
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(generate_all_responses_async(personas, questions, max_tokens, max_workers))
    except RuntimeError:
        # No running event loop, we can use asyncio.run directly
        return asyncio.run(generate_all_responses_async(personas, questions, max_tokens, max_workers))



# First, a quick test of the function using trader persona:
test_personas = {"trader": PERSONAS["trader"]}
test_questions = EVAL_QUESTIONS[:3]

test_responses = generate_all_responses(test_personas, test_questions)
print(f"Generated {len(test_responses)} responses:")

# Show a sample of the results:
for (persona_name, question), response in test_responses.items():
    response_sanitized = response.strip().replace("\n", "<br>")
    display(HTML(f"<details><summary><b>{persona_name}</b>: {question}</summary>{response_sanitized}</details>"))

# %%

# %%
# Once you've confirmed these work, run them all!
# Cache file for responses
RESPONSES_CACHE_FILE = Path(__file__).parent / "responses_cache.json"
RESPONSES_CACHE_FILE
# %%
def save_responses(responses: dict[tuple[str, str], str], filepath: Path):
    """Save responses to JSON file. Converts tuple keys to JSON strings."""
    # Use JSON encoding for keys to handle special characters in questions
    serializable = {json.dumps([persona, question]): resp for (persona, question), resp in responses.items()}
    filepath.write_text(json.dumps(serializable, indent=2))
    print(f"Saved {len(responses)} responses to {filepath}")

def load_responses(filepath: Path) -> dict[tuple[str, str], str]:
    """Load responses from JSON file. Converts JSON string keys back to tuples."""
    data = json.loads(filepath.read_text())
    return {tuple(json.loads(k)): v for k, v in data.items()}

# Check if cache exists, otherwise generate and save
if RESPONSES_CACHE_FILE.exists():
    print(f"Loading cached responses from {RESPONSES_CACHE_FILE}")
    responses = load_responses(RESPONSES_CACHE_FILE)
    print(f"Loaded {len(responses)} responses")
else:
    print("No cache found, generating all responses...")
    responses = generate_all_responses(PERSONAS, EVAL_QUESTIONS)
    save_responses(responses, RESPONSES_CACHE_FILE)
# %%
def format_messages(messages: list[dict[str, str]], tokenizer) -> tuple[str, int]:
    """Format a conversation for the model using its chat template.

    Args:
        messages: List of message dicts with "role" and "content" keys.
                 Can include "system", "user", and "assistant" roles.
        tokenizer: The tokenizer with chat template support

    Returns:
        full_prompt: The full formatted prompt as a string
        response_start_idx: The index of the first token in the last assistant message
    """
    # Apply chat template to get full conversation
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Get prompt without final assistant message to compute response_start_idx
    prompt_without_response = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    ).rstrip()

    response_start_idx = tokenizer(prompt_without_response, return_tensors="pt").input_ids.shape[1] + 1

    return full_prompt, response_start_idx
# %%
# %%
def extract_response_activations(
    model,
    tokenizer,
    system_prompts: list[str],
    questions: list[str],
    responses: list[str],
    layer: int,
) -> Float[Tensor, " num_examples d_model"]:
    """
    Extract mean activation over response tokens at a specific layer.

    Returns:
        Batch of mean activation vectors of shape (num_examples, hidden_size)
    """
    assert len(system_prompts) == len(questions) == len(responses)

    all_mean_activations = []
    for system_prompt, question, response in zip(system_prompts, questions, responses):
        print(question)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ]
        full_prompt, response_start_idx = format_messages(messages, tokenizer)
        tokens = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with t.inference_mode():
            outputs = model(**tokens, output_hidden_states=True)

        hidden_states = outputs.hidden_states[layer]  # (1, seq_len, d_model)

        seq_len = hidden_states.shape[1]
        response_mask = t.arange(seq_len, device=hidden_states.device) >= response_start_idx

        mean_activation = (hidden_states[0] * response_mask[:, None]).sum(0) / response_mask.sum()
        all_mean_activations.append(mean_activation.cpu())

    return t.stack(all_mean_activations)


test_activation = extract_response_activations(
    model=model,
    tokenizer=tokenizer,
    system_prompts=[PERSONAS["assistant"]],
    questions=EVAL_QUESTIONS[:1],
    responses=["I would suggest taking time to reflect on your goals and values."],
    layer=NUM_LAYERS // 2,
)
print(f"Extracted activation shape: {test_activation.shape}")
print(f"Activation norm: {test_activation.norm().item():.2f}")
# %%

def extract_response_activations_batched(
    model,
    tokenizer,
    system_prompts: list[str],
    questions: list[str],
    responses: list[str],
    layer: int,
    batch_size: int = 4,
) -> Float[Tensor, " num_examples d_model"]:
    """
    Extract mean activation over response tokens at a specific layer (batched version).
    Returns:
        Batch of mean activation vectors of shape (num_examples, hidden_size)
    """
    assert len(system_prompts) == len(questions) == len(responses)
# Build messages lists
    messages_list = [
        [
            {"role": "user", "content": f"{sp}\n\n{q}"},
            {"role": "assistant", "content": r},
        ]
        for sp, q, r in zip(system_prompts, questions, responses)
    ]
    formatted_messages = [format_messages(msgs, tokenizer) for msgs in messages_list]
    messages, response_start_indices = list(zip(*formatted_messages))
# Convert to lists for easier slicing
    messages = list(messages)
    response_start_indices = list(response_start_indices)
# Create list to store hidden states (as we iterate through batches)
    all_hidden_states: list[Float[Tensor, " num_examples d_model"]] = []
    idx = 0
    while idx < len(messages):
        # Tokenize the next batch of messages
        next_messages = messages[idx : idx + batch_size]
        next_indices = response_start_indices[idx : idx + batch_size]
        full_tokens = tokenizer(next_messages, return_tensors="pt", padding=True).to(model.device)

        with t.inference_mode():
            new_outputs = model(**full_tokens, output_hidden_states=True)

        batch_hidden_states = new_outputs.hidden_states[layer]  # (batch_size, seq_len, hidden_size)

        current_batch_size, seq_len, _ = batch_hidden_states.shape
        seq_pos_array = einops.repeat(t.arange(seq_len), "seq -> batch seq", batch=current_batch_size)
        model_response_mask = seq_pos_array >= t.tensor(next_indices)[:, None]
        model_response_mask = model_response_mask.to(batch_hidden_states.device)

        batch_mean_activation = (batch_hidden_states * model_response_mask[..., None]).sum(1) / model_response_mask.sum(
            1, keepdim=True
        )
        all_hidden_states.append(batch_mean_activation.cpu())
        
    # Concatenate all batches
        mean_activation = t.cat(all_hidden_states, dim=0)
        return mean_activation
# %%
def extract_persona_vectors(
    model,
    tokenizer,
    personas: dict[str, str],
    questions: list[str],
    responses: dict[tuple[str, str], str],
    layer: int,
    scores: dict[tuple[str, str], int] | None = None,
    score_threshold: int = 3,
) -> dict[str, Float[Tensor, " d_model"]]:
    """
    Extract mean activation vector for each persona.

    Args:
        model: The language model
        tokenizer: The tokenizer
        personas: Dict mapping persona name to system prompt
        questions: List of evaluation questions
        responses: Dict mapping (persona, question) to response text
        layer: Which layer to extract activations from
        scores: Optional dict mapping (persona, question) to judge score (0-3)
        score_threshold: Minimum score required to include response (default 3)

    Returns:
        Dict mapping persona name to mean activation vector
    """
    assert questions and personas and responses, "Invalid inputs"

    persona_vectors = {}

    for persona_name, system_prompt in personas.items():
        # Collect all responses for this persona
        system_prompts_list = []
        questions_list = []
        responses_list = []

        for question in questions:
            key = (persona_name, question)
            if key in responses:
                # Optionally filter by score
                if scores is None or scores.get(key, score_threshold) >= score_threshold:
                    system_prompts_list.append(system_prompt)
                    questions_list.append(question)
                    responses_list.append(responses[key])

        if not responses_list:
            print(f"Warning: No valid responses for persona '{persona_name}', skipping")
            continue

        # Extract activations for all responses of this persona
        activations = extract_response_activations(
            model, tokenizer, system_prompts_list, questions_list, responses_list, layer
        )

        # Average across all responses to get single persona vector
        persona_vectors[persona_name] = activations.mean(dim=0)
        print(f"Extracted vector for '{persona_name}' from {len(responses_list)} responses")

    return persona_vectors
# %%
# # Score all responses using the judge
# print("Scoring responses with LLM judge...")
# scores: dict[tuple[str, int], int] = {}

# for (persona_name, q_idx), response in tqdm(responses.items()):
#     if response:  # Skip empty responses
#         score = judge_role_response(
#             question=EVAL_QUESTIONS[q_idx],
#             response=response,
#             character=persona_name,
#         )
#         scores[(persona_name, q_idx)] = score
#         time.sleep(0.1)  # Rate limiting

# # Print filtering statistics per persona
# print("\nFiltering statistics (score >= 3 required):")
# for persona_name in PERSONAS.keys():
#     persona_scores = [scores.get((persona_name, q_idx), 0) for q_idx in range(len(EVAL_QUESTIONS))]
#     n_passed = sum(1 for s in persona_scores if s >= 3)
#     n_total = len(persona_scores)
#     print(f"  {persona_name}: {n_passed}/{n_total} passed ({n_passed / n_total:.0%})")

# Extract vectors (using the test subset from before)
EXTRACTION_LAYER = int(NUM_LAYERS * 0.65 + 0.5)  # 65% through the model
print(f"\nExtracting from layer {EXTRACTION_LAYER}")

# Cache file for persona vectors
PERSONA_VECTORS_CACHE_FILE = Path(__file__).parent / f"persona_vectors_layer{EXTRACTION_LAYER}.pt"

def save_persona_vectors(persona_vectors: dict[str, t.Tensor], filepath: Path):
    """Save persona vectors to a .pt file."""
    t.save(persona_vectors, filepath)
    print(f"Saved {len(persona_vectors)} persona vectors to {filepath}")

def load_persona_vectors(filepath: Path) -> dict[str, t.Tensor]:
    """Load persona vectors from a .pt file."""
    return t.load(filepath)

# Check if cache exists, otherwise extract and save
if PERSONA_VECTORS_CACHE_FILE.exists():
    print(f"Loading cached persona vectors from {PERSONA_VECTORS_CACHE_FILE}")
    persona_vectors = load_persona_vectors(PERSONA_VECTORS_CACHE_FILE)
    print(f"Loaded {len(persona_vectors)} persona vectors")
else:
    print("No cache found, extracting persona vectors...")
    persona_vectors = extract_persona_vectors(
        model=model,
        tokenizer=tokenizer,
        personas=PERSONAS,
        questions=EVAL_QUESTIONS,
        responses=responses,
        layer=EXTRACTION_LAYER,
    )
    save_persona_vectors(persona_vectors, PERSONA_VECTORS_CACHE_FILE)

print(f"\nExtracted vectors for {len(persona_vectors)} personas")
for name, vec in persona_vectors.items():
    print(f"  {name}: norm = {vec.norm().item():.2f}")
# %%
def compute_cosine_similarity_matrix(
    persona_vectors: dict[str, Float[Tensor, " d_model"]],
) -> tuple[Float[Tensor, "n_personas n_personas"], list[str]]:
    """
    Compute pairwise cosine similarity between persona vectors.
    Returns:
        Tuple of (similarity matrix, list of persona names in order)
    """
    names = list(persona_vectors.keys())
# Stack vectors into matrix
    vectors = t.stack([persona_vectors[name] for name in names])
# Normalize
    vectors_norm = vectors / vectors.norm(dim=1, keepdim=True)
# Compute cosine similarity
    cos_sim = vectors_norm @ vectors_norm.T
    return cos_sim, names

cos_sim_matrix, persona_names = compute_cosine_similarity_matrix(persona_vectors)

fig = px.imshow(
    cos_sim_matrix.float(),
    x=persona_names,
    y=persona_names,
    title="Persona Cosine Similarity Matrix (Uncentered)",
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
)
fig.update_layout(
    width=900,
    height=900,
    xaxis=dict(tickangle=45),
)
fig.show()
# %%
def compute_cosine_similarity_matrix_centered(
    persona_vectors: dict[str, Float[Tensor, " d_model"]],
) -> tuple[Float[Tensor, "n_personas n_personas"], list[str]]:
    """
    Compute pairwise cosine similarity between centered persona vectors.
    Returns:
        Tuple of (similarity matrix, list of persona names in order)
    """
    names = list(persona_vectors.keys())
# Stack vectors into matrix and center by subtracting mean
    vectors = t.stack([persona_vectors[name] for name in names])
    vectors = vectors - vectors.mean(dim=0)
# Normalize
    vectors_norm = vectors / vectors.norm(dim=1, keepdim=True)
# Compute cosine similarity
    cos_sim = vectors_norm @ vectors_norm.T
    return cos_sim, names

cos_sim_matrix_centered, persona_names = compute_cosine_similarity_matrix_centered(persona_vectors)

fig = px.imshow(
    cos_sim_matrix_centered.float(),
    x=persona_names,
    y=persona_names,
    title="Persona Cosine Similarity Matrix (Centered)",
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
)
fig.update_layout(
    width=900,
    height=900,
    xaxis=dict(tickangle=45),
)
fig.show()

# %%
# %%
def pca_decompose_persona_vectors(
    persona_vectors: dict[str, Float[Tensor, " d_model"]],
    default_personas: list[str] = DEFAULT_PERSONAS,
) -> tuple[
    Float[Tensor, " d_model"], np.ndarray, PCA, list[str], int, int, np.ndarray
]:
    """
    Analyze persona space structure.
    Args:
        persona_vectors: Dict mapping persona name to vector
        default_personas: List of persona names considered "default" (neutral assistant behavior)
    Returns:
        Tuple of:
        - assistant_axis: Normalized direction from role-playing toward default/assistant behavior
        - pca_coords: 2D PCA coordinates for each persona (n_personas, 2)
        - pca: Fitted PCA object, via the method PCA.fit_transform
        - n_components_90: Number of components to reach >=90% variance explained
        - n_components_99: Number of components to reach >=99% variance explained
        - cumulative_var: Cumulative explained variance per component
    """
    names = list(persona_vectors.keys())
    vectors = t.stack([persona_vectors[name] for name in names])
    # Compute Assistant Axis: mean(default) - mean(all_roles_excluding_default)
    # This points from role-playing behavior toward default assistant behavior
    default_vecs = [persona_vectors[name] for name in default_personas if name in persona_vectors]
    assert default_vecs, "Need at least some default vectors to subtract"
    mean_default = t.stack(default_vecs).mean(dim=0)
    # Get all personas excluding defaults
    role_names = [name for name in names if name not in default_personas]
    if role_names:
        role_vecs = t.stack([persona_vectors[name] for name in role_names])
        mean_roles = role_vecs.mean(dim=0)
    else:
        # Fallback if no roles
        mean_roles = vectors.mean(dim=0)
    assistant_axis = mean_default - mean_roles
    axis_norm = assistant_axis.norm()
    assert axis_norm > 0, "Assistant axis norm is zero; check persona vectors"
    assistant_axis = assistant_axis / axis_norm
    # PCA (sklearn expects CPU numpy arrays)
    vectors_np = vectors.detach().float().cpu().numpy()
    pca_full = PCA()
    pca_full.fit(vectors_np)
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_90 = int(np.searchsorted(cumulative_var, 0.90) + 1)
    n_components_99 = int(np.searchsorted(cumulative_var, 0.99) + 1)
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(vectors_np)
    return (
        assistant_axis,
        pca_coords,
        pca,
        names,
        n_components_90,
        n_components_99,
        cumulative_var,
    )


# Compute mean vector to handle constant vector problem (same as in centered cosine similarity)
# This will be subtracted from activations before projection to center around zero
persona_vectors = {k: v.float() for k, v in persona_vectors.items()}
mean_vector = t.stack(list(persona_vectors.values())).mean(dim=0)
persona_vectors_centered = {k: v - mean_vector for k, v in persona_vectors.items()}

# Perform PCA decomposition on space
(
    assistant_axis,
    pca_coords,
    pca,
    pca_names,
    n_components_90,
    n_components_99,
    cumulative_var,
) = pca_decompose_persona_vectors(persona_vectors_centered)
assistant_axis = assistant_axis.to(DEVICE, dtype=DTYPE)  # Keep model dtype for any downstream use
assistant_axis_plot = assistant_axis.float()  # Plotly doesn't support bfloat16

print(f"Assistant Axis norm: {assistant_axis.norm().item():.4f}")
print(
    f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}"
)
print(f"Components for >=90% variance: {n_components_90}")
print(f"Components for >=99% variance: {n_components_99}")

# Compute projection onto assistant axis for coloring
vectors = t.stack([persona_vectors_centered[name] for name in pca_names]).to(DEVICE, dtype=DTYPE)
# Normalize vectors before projecting (so projections are in [-1, 1] range)
vectors_normalized = vectors / vectors.norm(dim=1, keepdim=True)
projections = (vectors_normalized.float() @ assistant_axis_plot).cpu().numpy()

# 2D scatter plot
fig = px.scatter(
    x=pca_coords[:, 0],
    y=pca_coords[:, 1],
    text=pca_names,
    color=projections,
    color_continuous_scale="RdBu",
    title="Persona Space (PCA) colored by Assistant Axis projection",
    labels={
        "x": f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        "y": f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        "color": "Assistant Axis",
    },
)
fig.update_traces(textposition="top center", marker=dict(size=10))
fig.show()

# Components vs cumulative variance explained
components = np.arange(1, len(cumulative_var) + 1)
fig = px.line(
    x=components,
    y=cumulative_var,
    title="Cumulative Variance Explained by PCA Components",
    labels={"x": "Number of components", "y": "Cumulative variance explained"},
)
fig.add_hline(y=0.90, line_dash="dash", line_color="gray")
fig.add_hline(y=0.99, line_dash="dash", line_color="gray")
fig.show()
# %%
# %%
