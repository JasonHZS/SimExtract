"""Static files serving router."""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Adjust path: src/app/routers/static.py -> src/app/routers -> src/app -> src -> root
# Original path was src/static, so from src/app/routers we go up 3 levels to src/static?
# No, PROJECT_ROOT is in src/app/config.py, let's use that or relative path
# src/static is where the files are.
# From this file: ../../../src/static
# Or just absolute path via config.

from src.app.config import PROJECT_ROOT

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
