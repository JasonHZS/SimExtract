"""Attribution analysis router."""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends
from src.data_pipeline.samplers import RandomQuerySampler
from src.data_pipeline.stores.chroma_store import ChromaStore
from src.attribution.segmented import SegmentedAttribution
from src.attribution.token_wise import SparseAttribution, ColBERTAttribution

from src.server.schemas import AnalysisRequest, AnalysisResult, SpanInfo, KeywordInfo, SparseTokenInfo
from src.server.dependencies import (
    get_chroma_store,
    get_sparse_attribution,
    get_segmented_attribution,
    get_colbert_attribution
)

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/analyze_random", response_model=AnalysisResult)
async def analyze_random(
    request: AnalysisRequest = None,
    store: ChromaStore = Depends(get_chroma_store),
    sparse: SparseAttribution = Depends(get_sparse_attribution),
    segmented: Optional[SegmentedAttribution] = Depends(get_segmented_attribution),
    colbert: ColBERTAttribution = Depends(get_colbert_attribution),
):
    """Randomly sample a document pair and run attribution analysis.
    
    1. Samples a random document from the collection
    2. Gets its top-1 similar document
    3. Runs both Segmented and Sparse attribution
    4. Returns the top span from each method
    
    Args:
        request: Request body with collection name (required).
    """
    # Validate collection is provided
    if request is None or not request.collection:
        raise HTTPException(
            status_code=400,
            detail="Collection name is required. Please select a collection."
        )
    collection_name = request.collection
    
    # Validate collection exists
    collections = store.list_collections()
    if collection_name not in collections:
        raise HTTPException(
            status_code=400,
            detail=f"Collection '{collection_name}' not found. Available: {collections}"
        )
    
    # Sample and query
    sampler = RandomQuerySampler(store)
    try:
        results = sampler.sample_and_query(
            collection_name=collection_name,
            n_results=1  # Get top-1 similar (plus query itself = 2 docs)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sampling failed: {e}")
    
    ids = results["ids"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]
    
    if len(ids) < 2:
        raise HTTPException(
            status_code=500,
            detail="Not enough documents in collection for comparison"
        )
    
    source_id = ids[0]
    source_text = documents[0] or ""
    target_id = ids[1]
    target_text = documents[1] or ""
    similarity_distance = distances[1]
    
    if not source_text.strip() or not target_text.strip():
        raise HTTPException(status_code=500, detail="Empty document sampled")
    
    # Run attribution methods
    spans: List[SpanInfo] = []
    colbert_keywords: List[KeywordInfo] = []
    sparse_tokens: List[SparseTokenInfo] = []
    
    # 1. Sparse Attribution (always available since BGE-M3 is local)
    try:
        sparse_result = sparse.extract(source_text, target_text)
        if sparse_result.spans:
            top_span = sparse_result.spans[0]
            spans.append(SpanInfo(
                text=top_span.text,
                start_idx=top_span.start_idx,
                end_idx=top_span.end_idx,
                score=top_span.score,
                method="sparse",
            ))
        # Extract top contributing tokens for display (filter tokens < 2 chars)
        top_tokens = sparse_result.metadata.get("top_contributing_tokens", [])
        if top_tokens:
            filtered_tokens = [t for t in top_tokens if len(t.get("token", "")) >= 2]
            top_k = sparse.top_k_tokens  # 从配置读取关键词数量
            sparse_tokens = [
                SparseTokenInfo(
                    token=t.get("token", ""),
                    score=t.get("score", 0.0),
                    normalized_score=t.get("normalized_score", 0.0),
                )
                for t in filtered_tokens[:top_k]
            ]
    except Exception as e:
        logger.error(f"Sparse attribution failed: {e}")
    
    # 2. Segmented Attribution (may not be available if TEI is down)
    try:
        if segmented:
            segmented_result = segmented.extract(source_text, target_text)
            if segmented_result.spans:
                top_span = segmented_result.spans[0]
                spans.append(SpanInfo(
                    text=top_span.text,
                    start_idx=top_span.start_idx,
                    end_idx=top_span.end_idx,
                    score=top_span.score,
                    method="segmented",
                ))
    except Exception as e:
        logger.warning(f"Segmented attribution failed: {e}")
    
    # 3. ColBERT Attribution (uses BGE-M3 multi-vector late interaction)
    try:
        colbert_result = colbert.extract(source_text, target_text)
        if colbert_result.spans:
            top_span = colbert_result.spans[0]
            spans.append(SpanInfo(
                text=top_span.text,
                start_idx=top_span.start_idx,
                end_idx=top_span.end_idx,
                score=top_span.score,
                method="colbert",
            ))
        topic_keywords = colbert_result.metadata.get("topic_keywords", [])
        if topic_keywords:
            colbert_keywords = [
                KeywordInfo(
                    text=kw.get("text", ""),
                    start_idx=kw.get("start_idx", 0),
                    end_idx=kw.get("end_idx", 0),
                    score=kw.get("score", 0.0),
                )
                for kw in topic_keywords
            ]
    except Exception as e:
        logger.error(f"ColBERT attribution failed: {e}")
    
    return AnalysisResult(
        source_text=source_text,
        source_id=source_id,
        target_text=target_text,
        target_id=target_id,
        similarity_distance=similarity_distance,
        spans=spans,
        colbert_topic_keywords=colbert_keywords,
        sparse_top_tokens=sparse_tokens,
    )
