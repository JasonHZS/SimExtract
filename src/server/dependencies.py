"""Dependency injection for the server."""

import logging
from functools import lru_cache
from typing import Optional

from src.data_pipeline.stores.chroma_store import ChromaStore
from src.attribution.segmented import SegmentedAttribution
from src.attribution.token_wise import SparseAttribution, ColBERTAttribution
from src.server.config import CHROMA_DB_PATH, load_config

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_chroma_store() -> ChromaStore:
    """Get or create ChromaStore singleton."""
    logger.info(f"Initializing ChromaStore at {CHROMA_DB_PATH}")
    return ChromaStore(persist_directory=str(CHROMA_DB_PATH))


@lru_cache(maxsize=1)
def get_sparse_attribution() -> SparseAttribution:
    """Get or create SparseAttribution singleton (loads BGE-M3 once)."""
    config = load_config().get("sparse", {})
    logger.info("Initializing SparseAttribution (BGE-M3 model)...")
    return SparseAttribution(config)


@lru_cache(maxsize=1)
def get_segmented_attribution() -> Optional[SegmentedAttribution]:
    """Get or create SegmentedAttribution singleton.
    
    Returns None if TEI vectorizer is not available.
    """
    config = load_config().get("segmented", {})
    # Force fixed_length method for this demo
    config["segmentation_method"] = "fixed_length"
    
    try:
        logger.info("Initializing SegmentedAttribution (TEI vectorizer)...")
        return SegmentedAttribution(config)
    except Exception as e:
        logger.warning(f"SegmentedAttribution initialization failed: {e}")
        logger.warning("TEI vectorizer may not be running. Segmented method will be unavailable.")
        return None


@lru_cache(maxsize=1)
def get_colbert_attribution() -> ColBERTAttribution:
    """Get or create ColBERTAttribution singleton (uses BGE-M3 model).
    
    ColBERT uses the same BGE-M3 model as Sparse, but leverages multi-vector
    late interaction for span extraction.
    """
    config = load_config().get("colbert", {})
    logger.info("Initializing ColBERTAttribution (BGE-M3 ColBERT vectors)...")
    return ColBERTAttribution(config)
