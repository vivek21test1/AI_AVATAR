"""
Models router — controller for all /models endpoints.

Responsibilities (controller only):
  • Parse and validate path parameters
  • Delegate to services
  • Build HTTP responses from schema objects
  • Raise HTTPException only for non-domain errors (should be rare)

All business logic lives in the services.
"""

from fastapi import APIRouter, BackgroundTasks, Depends

from app.core.exceptions import ModelNotFoundError
from app.core.registry import MODEL_REGISTRY
from app.dependencies import get_download_service, get_model_repository
from app.repositories.model_repository import ModelRepository
from app.schemas.model_schema import DownloadTriggerResponse, ModelInfoResponse
from app.services.model_download_service import ModelDownloadService

router = APIRouter(prefix="/models", tags=["Models"])


# ---------------------------------------------------------------------------
# List all models
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=list[ModelInfoResponse],
    summary="List all supported models",
)
def list_models(
    model_repo: ModelRepository = Depends(get_model_repository),
    dl_service: ModelDownloadService = Depends(get_download_service),
):
    """Return every registered model with its local download status."""
    return [
        _build_info(name, model_repo, dl_service)
        for name in MODEL_REGISTRY
    ]


# ---------------------------------------------------------------------------
# Single model info
# ---------------------------------------------------------------------------

@router.get(
    "/{model_name}",
    response_model=ModelInfoResponse,
    summary="Get model details",
)
def get_model(
    model_name: str,
    model_repo: ModelRepository = Depends(get_model_repository),
    dl_service: ModelDownloadService = Depends(get_download_service),
):
    """Detailed information and local download status for one model."""
    _require_known(model_name)
    return _build_info(model_name, model_repo, dl_service)


# ---------------------------------------------------------------------------
# Trigger download
# ---------------------------------------------------------------------------

@router.post(
    "/{model_name}/download",
    status_code=202,
    response_model=DownloadTriggerResponse,
    summary="Trigger model download",
)
def trigger_download(
    model_name: str,
    background_tasks: BackgroundTasks,
    model_repo: ModelRepository = Depends(get_model_repository),
    dl_service: ModelDownloadService = Depends(get_download_service),
):
    """
    Start downloading model weights + GitHub repo in the background.
    Returns immediately; poll `GET /models/{model_name}` to track readiness.
    """
    _require_known(model_name)

    if model_repo.is_ready(model_name):
        return DownloadTriggerResponse(
            model_name=model_name,
            started=False,
            message=f"'{model_name}' is already downloaded and ready.",
        )

    started = dl_service.start_background_download(model_name)
    return DownloadTriggerResponse(
        model_name=model_name,
        started=started,
        message=(
            f"Download started for '{model_name}'."
            if started
            else f"Download already in progress for '{model_name}'."
        ),
    )


# ---------------------------------------------------------------------------
# Download status
# ---------------------------------------------------------------------------

@router.get(
    "/{model_name}/download/status",
    response_model=ModelInfoResponse,
    summary="Check download status",
)
def download_status(
    model_name: str,
    model_repo: ModelRepository = Depends(get_model_repository),
    dl_service: ModelDownloadService = Depends(get_download_service),
):
    """Return the current download/readiness status for one model."""
    _require_known(model_name)
    return _build_info(model_name, model_repo, dl_service)


# ---------------------------------------------------------------------------
# Helpers (private to this module)
# ---------------------------------------------------------------------------

def _require_known(model_name: str) -> None:
    if model_name not in MODEL_REGISTRY:
        raise ModelNotFoundError(model_name)


def _build_info(
    name: str,
    model_repo: ModelRepository,
    dl_service: ModelDownloadService,
) -> ModelInfoResponse:
    cfg = MODEL_REGISTRY[name]
    return ModelInfoResponse(
        model_name=name,
        display_name=cfg["display_name"],
        hf_repo=cfg["hf_repo"],
        vram_gb=cfg["vram_gb"],
        output_format=cfg["output_format"],
        description=cfg["description"],
        license=cfg["license"],
        commercial_use=cfg["commercial_use"],
        weights_downloaded=model_repo.is_weights_downloaded(name),
        repo_cloned=model_repo.is_repo_cloned(name),
        ready=model_repo.is_ready(name),
        download_in_progress=dl_service.is_downloading(name),
    )
