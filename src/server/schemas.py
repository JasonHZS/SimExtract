"""Pydantic models for API responses and requests."""

from typing import List, Optional
from pydantic import BaseModel


class SpanInfo(BaseModel):
    """Attribution span information."""
    text: str
    start_idx: int
    end_idx: int
    score: float
    method: str


class AnalysisRequest(BaseModel):
    """Request for attribution analysis."""
    collection: Optional[str] = None


class AnalysisResult(BaseModel):
    """Result of attribution analysis."""
    source_text: str
    source_id: str
    target_text: str
    target_id: str
    similarity_distance: float
    spans: List[SpanInfo]


class CollectionInfo(BaseModel):
    """Information about a single collection."""
    name: str
    count: int


class CollectionsResponse(BaseModel):
    """Response for listing collections."""
    collections: List[CollectionInfo]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    sparse_loaded: bool
    segmented_loaded: bool
    total_collections: int
