"""Common utilities for token-wise attribution methods."""

from typing import Any, Dict, List


def tokenize_with_positions(tokenizer, text: str) -> List[Dict[str, Any]]:
    """Tokenize text and get character positions for each token.

    Args:
        tokenizer: HuggingFace tokenizer instance
        text: Text to tokenize

    Returns:
        List of token info dictionaries with keys:
            - token_id: int, the token ID
            - token: str, the token string
            - start: int, character start position
            - end: int, character end position
            - index: int, position in the token sequence
    """
    # Encode with offset mapping
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    tokens = []
    for idx, (token_id, (start, end)) in enumerate(
        zip(encoding['input_ids'], encoding['offset_mapping'])
    ):
        # Skip special tokens or empty spans
        if start == end:
            continue

        token_str = tokenizer.decode([token_id])
        tokens.append({
            "token_id": token_id,
            "token": token_str,
            "start": start,
            "end": end,
            "index": idx
        })

    return tokens


def normalize_score(score: float, max_score: float) -> float:
    """Normalize score to [0, 1] range.

    Args:
        score: Raw score to normalize
        max_score: Maximum score for normalization

    Returns:
        Normalized score in [0, 1]
    """
    if max_score <= 0:
        return 0.0
    normalized = score / max_score
    return max(0.0, min(1.0, normalized))

