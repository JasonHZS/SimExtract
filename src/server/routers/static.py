"""Static files serving router."""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Adjust path: src/server/routers/static.py -> src/server/routers -> src/server -> src -> root
# Original path was src/static, so from src/server/routers we go up 3 levels to src/static?
# No, PROJECT_ROOT is in src/server/config.py, let's use that or relative path
# src/static is where the files are.
# From this file: ../../../src/static
# Or just absolute path via config.

from src.server.config import PROJECT_ROOT

router = APIRouter()

STATIC_DIR = PROJECT_ROOT / "src" / "static"

def mount_static(app):
    """Mount static files to the app."""
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@router.get("/")
async def serve_frontend():
    """Serve the frontend HTML page."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)
