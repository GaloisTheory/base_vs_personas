"""
Transcript Projection Module

Project multi-turn conversation transcripts onto persona axes.
Extracts hidden state activations from assistant turns and computes
dot-product projections onto a given direction vector (e.g. assistant axis).

Usage:
    from utils.transcript_projection import project_transcript, discover_transcripts

    model = AutoModelForCausalLM.from_pretrained(...)
    tokenizer = AutoTokenizer.from_pretrained(...)
    axis = ...  # (d_model,) direction vector

    projections = project_transcript(model, tokenizer, conversation, axis)
    # projections: np.ndarray of shape (n_assistant_turns,)
"""

import json
from pathlib import Path

import numpy as np
import torch


def discover_transcripts(base_dir: Path) -> dict[str, list[dict]]:
    """Find all transcript JSON files and return {label: conversation_list}.

    Walks ``base_dir`` recursively for ``.json`` files and builds short labels:
      - ``case_studies/llama-3.3-70b/X.json`` → ``"llama-70b/X"``
      - ``case_studies/qwen-3-32b/X.json``    → ``"qwen-32b/X"``
      - ``persona_drift/X.json``              → ``"drift/X"``

    Handles the ``.json.json`` double-extension edge case.
    """
    transcripts: dict[str, list[dict]] = {}
    for json_path in sorted(base_dir.rglob("*.json")):
        rel = json_path.relative_to(base_dir)
        parts = list(rel.parts)
        # Strip .json extension (handle .json.json edge case)
        stem = parts[-1]
        while stem.endswith(".json"):
            stem = stem[:-5]
        parts[-1] = stem

        if parts[0] == "case_studies" and len(parts) >= 3:
            model_dir = parts[1]
            if "llama" in model_dir:
                model_short = "llama-70b"
            elif "qwen" in model_dir:
                model_short = "qwen-32b"
            else:
                model_short = model_dir
            label = f"{model_short}/{parts[-1]}"
        elif parts[0] == "persona_drift":
            label = f"drift/{parts[-1]}"
        else:
            label = "/".join(parts)

        with open(json_path) as f:
            data = json.load(f)
        transcripts[label] = data["conversation"]
    return transcripts


def format_conversation_raw(conversation: list[dict]) -> str:
    """Format multi-turn conversation as raw plaintext.

    Matches the raw format used for persona vector extraction:
        System: \\n\\nUser: ...\\n\\nAssistant: ...\\n\\n...

    Includes empty System: prefix to match persona vector extraction format
    (persona_vectors.py:_format_messages_raw always prepends "System: {system}").
    """
    parts = ["System: "]  # empty system prompt to match extraction format
    for msg in conversation:
        role = msg["role"].capitalize()
        parts.append(f"{role}: {msg['content']}")
    return "\n\n".join(parts)


def find_assistant_spans(
    tokenizer,
    conversation: list[dict],
    format_mode: str = "raw",
) -> list[tuple[int, int]]:
    """Find token spans for each assistant turn's *response content* only.

    Args:
        tokenizer: HuggingFace tokenizer
        conversation: List of message dicts with "role" and "content" keys
        format_mode: "raw" (structured plaintext) or "chat" (chat template)

    Returns:
        List of (start_idx, end_idx) token positions covering response
        content only (excludes separator and role label tokens).
    """
    if format_mode == "raw":
        return _find_assistant_spans_raw(tokenizer, conversation)
    elif format_mode == "chat":
        return _find_assistant_spans_chat(tokenizer, conversation)
    else:
        raise ValueError(f"Unknown format_mode: {format_mode!r}. Use 'raw' or 'chat'.")


def _find_assistant_spans_raw(
    tokenizer,
    conversation: list[dict],
) -> list[tuple[int, int]]:
    """Find assistant spans using raw plaintext formatting.

    For each assistant turn i:
    - prefix = conversation[:i] formatted + "\\n\\nAssistant:" (the label)
    - response_start = len(tokenize(prefix)) + 1 (space after colon)
    - response_end = len(tokenize(conversation[:i+1] formatted))
    """
    spans = []
    for i, msg in enumerate(conversation):
        if msg["role"] != "assistant":
            continue

        # Text up to and including this assistant turn
        through_text = format_conversation_raw(conversation[: i + 1])
        through_tokens = tokenizer(through_text, return_tensors="pt").input_ids.shape[1]

        # Text up to the "Assistant:" label (excluding response content).
        prefix_text = format_conversation_raw(conversation[:i])
        prompt_up_to_label = prefix_text + "\n\nAssistant:"
        label_tokens = tokenizer(prompt_up_to_label, return_tensors="pt").input_ids.shape[1]

        # +1 for the space after "Assistant:" before actual response content
        response_start = label_tokens + 1
        spans.append((response_start, through_tokens))

    return spans


def _find_assistant_spans_chat(
    tokenizer,
    conversation: list[dict],
) -> list[tuple[int, int]]:
    """Find assistant spans using chat template formatting.

    Mirrors persona_vectors.py:_format_messages:
    - Full text = apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    - For each assistant turn i:
        prefix = apply_chat_template(conversation[:i], add_generation_prompt=True).rstrip()
        response_start = len(tokenize(prefix)) + 1
        response_end = len(tokenize(full_through_turn_i))
    """
    spans = []
    for i, msg in enumerate(conversation):
        if msg["role"] != "assistant":
            continue

        # Full text through this assistant turn
        through_text = tokenizer.apply_chat_template(
            conversation[: i + 1], tokenize=False, add_generation_prompt=False
        )
        through_tokens = tokenizer(through_text, return_tensors="pt").input_ids.shape[1]

        # Prompt up to where this assistant response starts
        prefix_text = tokenizer.apply_chat_template(
            conversation[:i], tokenize=False, add_generation_prompt=True
        ).rstrip()
        prefix_tokens = tokenizer(prefix_text, return_tensors="pt").input_ids.shape[1]

        # +1 for the token after the generation prompt
        response_start = prefix_tokens + 1
        spans.append((response_start, through_tokens))

    return spans


def project_transcript(
    model,
    tokenizer,
    conversation: list[dict],
    axis: torch.Tensor,
    format_mode: str = "raw",
    layer: int = 21,
    max_seq_len: int = 8192,
) -> np.ndarray:
    """Run transcript through model and project assistant turn activations onto axis.

    Args:
        model: HuggingFace causal LM (already loaded on GPU)
        tokenizer: Corresponding tokenizer
        conversation: List of message dicts with "role" and "content" keys
        axis: Direction vector of shape (d_model,) to project onto
        format_mode: "raw" (structured plaintext) or "chat" (chat template)
        layer: Which layer to extract hidden states from
        max_seq_len: Truncate input if longer than this

    Returns:
        np.ndarray of shape (n_assistant_turns,) with projection values.
        Model is NOT freed — caller manages lifecycle.
    """
    # Format full conversation
    if format_mode == "chat":
        full_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
    else:
        full_text = format_conversation_raw(conversation)

    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt")
    seq_len = inputs.input_ids.shape[1]
    print(f"  Sequence length: {seq_len} tokens")

    if seq_len > max_seq_len:
        print(f"  WARNING: Truncating from {seq_len} to {max_seq_len} tokens")
        inputs = {k: v[:, :max_seq_len] for k, v in inputs.items()}
        seq_len = max_seq_len

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Find assistant turn spans
    spans = find_assistant_spans(tokenizer, conversation, format_mode=format_mode)
    print(f"  Assistant turn spans: {len(spans)}")

    # Single forward pass
    print(f"  Running forward pass...")
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract hidden states at target layer
    hidden_states = outputs.hidden_states[layer]  # (1, seq_len, d_model)
    hidden_states = hidden_states[0].float()  # (seq_len, d_model) in float32

    # Project each assistant turn's mean activation onto the axis
    axis_device = axis.to(hidden_states.device)
    projections = []
    for turn_idx, (start, end) in enumerate(spans):
        if start >= seq_len:
            print(f"  Turn {turn_idx}: SKIPPED (start={start} >= seq_len={seq_len})")
            break
        end = min(end, seq_len)
        span_hidden = hidden_states[start:end]  # (span_len, d_model)
        mean_activation = span_hidden.mean(dim=0)  # (d_model,)
        proj = torch.dot(mean_activation, axis_device).item()
        projections.append(proj)
        print(f"  Turn {turn_idx}: tokens [{start}:{end}] ({end - start} tokens), projection={proj:.4f}")

    return np.array(projections)
