from fastapi import APIRouter

from app.core.registry import MODEL_REGISTRY

router = APIRouter(tags=["Health"])


@router.get("/", summary="Health check")
def health_check():
    """Service liveness check — returns status and the list of supported models."""
    return {
        "service": "AI Avatar Generation API",
        "status": "running",
        "supported_models": list(MODEL_REGISTRY.keys()),
        "docs": "/docs",
    }
