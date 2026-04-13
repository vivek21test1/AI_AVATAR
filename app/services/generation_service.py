"""
GenerationService — orchestrates avatar generation requests.

Sits between the router and JobService; it handles:
  • Validating that the model exists
  • Optionally triggering a background download before queueing the job
  • Saving the uploaded image to disk
  • Returning a JobResponse the router can serialise

This separation means the router contains zero business logic.
"""

import logging
import tempfile
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import ModelNotDownloadedError, ModelNotFoundError
from app.repositories.model_repository import ModelRepository
from app.schemas.job_schema import JobResponse
from app.services.job_service import JobService
from app.services.model_download_service import ModelDownloadService

logger = logging.getLogger(__name__)


class GenerationService:

    def __init__(
        self,
        job_service: JobService,
        model_repo: ModelRepository,
        download_service: ModelDownloadService,
    ) -> None:
        self._job_service   = job_service
        self._model_repo    = model_repo
        self._dl_service    = download_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        model_name: str,
        image_bytes: bytes,
        file_extension: str,
        auto_download: bool = True,
    ) -> JobResponse:
        """
        Validate → (optionally trigger download) → persist image → queue job.
        Returns the JobResponse for the newly created job.
        """
        self._validate_model(model_name, auto_download)
        image_path = self._persist_image(image_bytes, file_extension)
        return self._job_service.create_and_submit(
            model_name=model_name,
            image_path=image_path,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_model(self, model_name: str, auto_download: bool) -> None:
        if not self._model_repo.exists(model_name):
            raise ModelNotFoundError(model_name)

        if not self._model_repo.is_ready(model_name):
            if not auto_download:
                raise ModelNotDownloadedError(model_name)
            # auto_download=True: the job thread will handle downloading;
            # but we can also kick off a background download eagerly here
            # so it starts immediately rather than waiting for a thread slot.
            logger.info(
                f"[{model_name}] Not downloaded yet — job thread will download on first run"
            )

    def _persist_image(self, data: bytes, extension: str) -> str:
        """Write image bytes to a temp file and return the path."""
        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
            dir=settings.OUTPUT_DIR,
            prefix="upload_",
        )
        tmp.write(data)
        tmp.flush()
        tmp.close()
        return tmp.name
