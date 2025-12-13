"""Common utilities for token-wise attribution methods."""


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

