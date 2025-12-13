"""Health check router."""

import logging
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


@router.get("/collections", response_model=CollectionsResponse)
async def list_collections(
    store: ChromaStore = Depends(get_chroma_store),
):
    """List all available collections in ChromaDB."""
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
    
    return CollectionsResponse(collections=collections)


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
