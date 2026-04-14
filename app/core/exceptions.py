"""
Domain exceptions and their HTTP mappings.

Raise these from services/repositories; the exception handlers registered
in app/main.py convert them to proper HTTP responses automatically, so
routers never need to catch-and-re-raise as HTTPException.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------

class ModelNotFoundError(Exception):
    def __init__(self, model_name: str) -> None:
        super().__init__(f"Unknown model '{model_name}'")
        self.model_name = model_name


class ModelNotDownloadedError(Exception):
    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"Model '{model_name}' is not downloaded. "
            "Call POST /api/v1/models/{model_name}/download first, "
            "or set auto_download=true in the generate request."
        )
        self.model_name = model_name


class ModelDownloadError(Exception):
    """Raised when a HuggingFace download or git clone fails."""


class JobNotFoundError(Exception):
    def __init__(self, job_id: str) -> None:
        super().__init__(f"Job '{job_id}' not found")
        self.job_id = job_id


class JobNotReadyError(Exception):
    """Raised when trying to download a result that isn't finished yet."""
    def __init__(self, job_id: str, status: str) -> None:
        super().__init__(
            f"Job '{job_id}' is not complete yet (status={status}). "
            "Poll GET /api/v1/jobs/{job_id} until status is 'completed'."
        )
        self.job_id = job_id
        self.status = status


class GenerationError(Exception):
    """Raised when model inference fails."""


class ModelEnvironmentError(Exception):
    """Host cannot satisfy hardware / runtime needs for a model (e.g. CUDA-only)."""

    def __init__(self, model_name: str, detail: str) -> None:
        super().__init__(detail)
        self.model_name = model_name


# ---------------------------------------------------------------------------
# HTTP exception handlers
# ---------------------------------------------------------------------------

def register_exception_handlers(app: FastAPI) -> None:
    """Attach all domain-exception → HTTP handlers to the app."""

    @app.exception_handler(ModelNotFoundError)
    async def _model_not_found(_: Request, exc: ModelNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ModelNotDownloadedError)
    async def _model_not_downloaded(_: Request, exc: ModelNotDownloadedError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ModelDownloadError)
    async def _download_error(_: Request, exc: ModelDownloadError):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(JobNotFoundError)
    async def _job_not_found(_: Request, exc: JobNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(JobNotReadyError)
    async def _job_not_ready(_: Request, exc: JobNotReadyError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(GenerationError)
    async def _generation_error(_: Request, exc: GenerationError):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(ModelEnvironmentError)
    async def _model_environment(_: Request, exc: ModelEnvironmentError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})
