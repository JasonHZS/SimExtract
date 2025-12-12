"""Main entry point for the application."""

import logging
import sys  # <--- 新增
from contextlib import asynccontextmanager

# <--- 新增/修改：在导入 FastAPI 之前配置日志，确保输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

from fastapi import FastAPI
from src.server.routers import health, analysis, static
from src.server.dependencies import (
    get_chroma_store,
    get_sparse_attribution,
    get_segmented_attribution
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: load models at startup."""
    # Log device config at startup.
    try:
        from src.server.config import load_config
        config = load_config()
        sparse_config = config.get("sparse", {})
        device = sparse_config.get("device")
        if device is not None:
            logger.info(f"SparseAttribution configured device: {device}")
    except Exception as e:
        logger.warning(f"Failed to read device config: {e}")

    logger.info("=" * 60)
    logger.info("SimExtract Demo Server starting...")
    logger.info("=" * 60)
    
    # Pre-load all singletons at startup
    try:
        store = get_chroma_store()
        collections = store.list_collections()
        logger.info(f"ChromaStore ready. Collections: {collections}")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaStore: {e}")
    
    try:
        sparse = get_sparse_attribution()
        logger.info(f"SparseAttribution ready: {sparse.model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize SparseAttribution: {e}")
    
    try:
        segmented = get_segmented_attribution()
        if segmented:
            logger.info("SegmentedAttribution ready")
        else:
            logger.warning("SegmentedAttribution not available (TEI not running?)")
    except Exception as e:
        logger.warning(f"SegmentedAttribution initialization skipped: {e}")
    
    logger.info("=" * 60)
    logger.info("Server ready! Visit http://localhost:8001")
    logger.info("=" * 60)
    
    yield
    
    logger.info("Server shutting down...")


app = FastAPI(
    title="SimExtract Attribution Demo",
    description="Compare Segmented and Sparse attribution methods",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(analysis.router, prefix="/api", tags=["analysis"])
app.include_router(static.router, tags=["frontend"])

# Mount static files
static.mount_static(app)

if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    uvicorn.run(
        "src.server.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
