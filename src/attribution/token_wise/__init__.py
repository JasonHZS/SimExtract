"""Token-wise attribution methods package.

This package provides token-level attribution methods for analyzing
which tokens in text_b contribute most to its similarity with text_a.

Available methods:
- SparseAttribution: Uses BGE-M3 lexical weights for sparse token matching
- ColBERTAttribution: Uses ColBERT-style late interaction with MaxSim scoring
"""

from .sparse import SparseAttribution
from .colbert import ColBERTAttribution

__all__ = ["SparseAttribution", "ColBERTAttribution"]
