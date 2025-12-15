"""Health check router."""

import logging
from functools import lru_cache
from typing import List

from fastapi import APIRouter, Depends
from src.app.schemas import HealthResponse, CollectionsResponse, CollectionInfo
from src.app.dependencies import (
    get_chroma_store,
    get_sparse_attribution,
    get_segmented_attribution,
    get_colbert_attribution
)
from src.data_pipeline.stores.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

router = APIRouter()


@lru_cache(maxsize=1)
def _get_collections_info_cached(store_id: int) -> List[CollectionInfo]:
    """Cached helper function to get collection information.

    This function caches collection metadata to avoid repeated count() calls
    which can be slow for large collections. The cache is keyed by the store's
    identity (id(store)) and persists for the lifetime of the process.

    To clear the cache:
    - Restart the server
    - Call _get_collections_info_cached.cache_clear()
    - Use the /collections/refresh endpoint

    Args:
        store_id: Identity hash of the ChromaStore instance (from id(store))

    Returns:
        List of CollectionInfo objects with name and count
    """
    from src.app.dependencies import get_chroma_store
    store = get_chroma_store()

    collection_names = store.list_collections()
    collections: List[CollectionInfo] = []

    for name in collection_names:
        try:
            collection = store.get_collection(name)
            count = collection.count() if collection else 0
        except Exception as e:
            logger.warning(f"Failed to get count for collection '{name}': {e}")
            count = 0

        collections.append(CollectionInfo(name=name, count=count))

    # Sort by count descending (most populated first)
    collections.sort(key=lambda c: c.count, reverse=True)

    return collections


@router.get("/collections", response_model=CollectionsResponse)
async def list_collections(
    store: ChromaStore = Depends(get_chroma_store),
):
    """List all available collections in ChromaDB.

    This endpoint uses caching to improve performance. Collection counts are
    cached after the first request and reused for subsequent requests until
    the server is restarted or the cache is manually cleared.
    """
    collections = _get_collections_info_cached(id(store))
    return CollectionsResponse(collections=collections)


@router.post("/collections/refresh")
async def refresh_collections_cache():
    """Clear the collections cache to force refresh.

    Use this endpoint after adding/removing documents from collections
    to ensure the collection counts are up-to-date.

    Returns:
        Status message confirming cache was cleared
    """
    _get_collections_info_cached.cache_clear()
    logger.info("Collections cache cleared")
    return {"status": "ok", "message": "Collections cache cleared successfully"}


@router.get("/health", response_model=HealthResponse)
async def health_check(
    store: ChromaStore = Depends(get_chroma_store),
):
    """Check server health and loaded components."""
    # Count total collections
    collections = store.list_collections()
    total_collections = len(collections)
    
    sparse_loaded = False
    try:
        # We access the singleton directly to check if it's loaded without re-instantiating
        # if dependencies cache logic is used, calling the dependency function is fine
        from src.app.dependencies import get_sparse_attribution
        sparse = get_sparse_attribution()
        sparse_loaded = sparse._model is not None
    except Exception:
        pass
    
    segmented_loaded = False
    try:
        from src.app.dependencies import get_segmented_attribution
        segmented = get_segmented_attribution()
        segmented_loaded = segmented is not None
    except Exception:
        pass
    
    colbert_loaded = False
    try:
        from src.app.dependencies import get_colbert_attribution
        colbert = get_colbert_attribution()
        colbert_loaded = colbert._model is not None
    except Exception:
        pass
    
    return HealthResponse(
        status="ok",
        sparse_loaded=sparse_loaded,
        segmented_loaded=segmented_loaded,
        colbert_loaded=colbert_loaded,
        total_collections=total_collections,
    )
