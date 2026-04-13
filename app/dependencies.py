"""
Dependency injection providers.

All service/repository singletons are created once here and injected into
routers via FastAPI's Depends() mechanism.  Routers never instantiate
services directly — they only declare what they need.

To swap an implementation (e.g. replace in-memory JobRepository with a
Redis-backed one), change this file only; no router changes required.
"""

from app.repositories.job_repository import JobRepository
from app.repositories.model_repository import ModelRepository
from app.services.generation_service import GenerationService
from app.services.job_service import JobService
from app.services.model_download_service import ModelDownloadService

# ---------------------------------------------------------------------------
# Singletons — created once for the lifetime of the process
# ---------------------------------------------------------------------------

_model_repo   = ModelRepository()
_job_repo     = JobRepository()
_dl_service   = ModelDownloadService(model_repo=_model_repo)
_job_service  = JobService(job_repo=_job_repo, download_service=_dl_service)
_gen_service  = GenerationService(
    job_service=_job_service,
    model_repo=_model_repo,
    download_service=_dl_service,
)

# ---------------------------------------------------------------------------
# Provider functions (used with Depends())
# ---------------------------------------------------------------------------

def get_model_repository() -> ModelRepository:
    return _model_repo


def get_job_repository() -> JobRepository:
    return _job_repo


def get_download_service() -> ModelDownloadService:
    return _dl_service


def get_job_service() -> JobService:
    return _job_service


def get_generation_service() -> GenerationService:
    return _gen_service
