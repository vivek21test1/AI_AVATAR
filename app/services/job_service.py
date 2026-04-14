"""
JobService — business logic for the job lifecycle.

Responsibilities:
  • Create jobs and immediately submit them to a background thread
  • Execute the full pipeline: (optional download) → inference → ZIP
  • Expose CRUD operations that the controller (router) uses
  • Convert JobEntity → JobResponse for the API layer
"""

import logging
import shutil
import threading
import zipfile
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import (
    GenerationError,
    JobNotFoundError,
    JobNotReadyError,
)
from app.repositories.job_repository import JobEntity, JobRepository
from app.schemas.job_schema import JobResponse, JobStatus
from app.services.model_download_service import ModelDownloadService

logger = logging.getLogger(__name__)


class JobService:

    def __init__(
        self,
        job_repo: JobRepository,
        download_service: ModelDownloadService,
    ) -> None:
        self._job_repo = job_repo
        self._dl_service = download_service

    # ------------------------------------------------------------------
    # Public API (used by routers)
    # ------------------------------------------------------------------

    def create_and_submit(self, model_name: str, image_path: str) -> JobResponse:
        entity = self._job_repo.create(model_name=model_name, image_path=image_path)
        thread = threading.Thread(
            target=self._execute,
            args=(entity.id,),
            daemon=True,
        )
        thread.start()
        return self._to_response(entity)

    def get(self, job_id: str) -> JobResponse:
        entity = self._job_repo.get(job_id)
        if entity is None:
            raise JobNotFoundError(job_id)
        return self._to_response(entity)

    def get_entity(self, job_id: str) -> JobEntity:
        """Return the raw entity (used by the router for file-streaming)."""
        entity = self._job_repo.get(job_id)
        if entity is None:
            raise JobNotFoundError(job_id)
        return entity

    def list_all(self) -> list[JobResponse]:
        return [self._to_response(e) for e in self._job_repo.list_all()]

    def delete(self, job_id: str) -> None:
        entity = self._job_repo.get(job_id)
        if entity is None:
            raise JobNotFoundError(job_id)

        # Remove output files from disk
        for path in entity.output_files:
            Path(path).unlink(missing_ok=True)
        if entity.zip_path:
            Path(entity.zip_path).unlink(missing_ok=True)

        self._job_repo.delete(job_id)

    def assert_result_ready(self, job_id: str) -> JobEntity:
        """
        Raise appropriate domain exceptions if the job result is not available.
        Returns the entity when everything is fine.
        """
        entity = self.get_entity(job_id)

        if entity.status == JobStatus.FAILED:
            raise GenerationError(f"Job '{job_id}' failed: {entity.error}")

        if entity.status != JobStatus.COMPLETED or entity.zip_path is None:
            raise JobNotReadyError(job_id, entity.status.value)

        if not Path(entity.zip_path).exists():
            raise GenerationError(
                f"Result ZIP for job '{job_id}' is no longer on disk"
            )

        return entity

    # ------------------------------------------------------------------
    # Execution (runs in background thread)
    # ------------------------------------------------------------------

    def _execute(self, job_id: str) -> None:
        # Late import to break circular dependency with avatar_models.factory
        from app.avatar_models.factory import ModelFactory

        def update(**fields):
            self._job_repo.update(job_id, **fields)

        try:
            entity = self._job_repo.get(job_id)

            # ── Step 1: ensure model is downloaded ─────────────────────
            if not self._dl_service._model_repo.is_ready(entity.model_name):
                update(
                    status=JobStatus.DOWNLOADING,
                    message="Downloading model weights …",
                    progress=0.05,
                )
                self._dl_service.download(
                    entity.model_name,
                    progress_callback=lambda msg: update(message=msg),
                )

            # ── Step 2: run inference ───────────────────────────────────
            update(
                status=JobStatus.PROCESSING,
                message="Running inference …",
                progress=0.30,
            )

            out_dir = settings.OUTPUT_DIR / job_id
            out_dir.mkdir(parents=True, exist_ok=True)

            model = ModelFactory.get(entity.model_name)
            result = model.generate(
                image_path=entity.image_path,
                output_dir=str(out_dir),
            )

            output_files: list[str] = result.get("output_files", [])
            output_files = self._output_files_with_source_image(
                out_dir, entity.image_path, output_files
            )
            update(output_files=output_files, progress=0.85, message="Packaging output …")

            # ── Step 3: ZIP results ─────────────────────────────────────
            zip_path = str(out_dir / "result.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for fp in output_files:
                    zf.write(fp, arcname=Path(fp).name)

            update(
                status=JobStatus.COMPLETED,
                zip_path=zip_path,
                progress=1.0,
                message="Done",
            )
            logger.info(f"Job {job_id} completed → {zip_path}")

        except Exception as exc:
            logger.exception(f"Job {job_id} failed")
            update(status=JobStatus.FAILED, error=str(exc), message="Job failed")

    # ------------------------------------------------------------------
    # Packaging helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _output_files_with_source_image(
        out_dir: Path,
        image_path: str,
        output_files: list[str],
    ) -> list[str]:
        """
        Copy the API upload into the job output dir as ``uploaded_input.*`` and
        ensure that path is listed first so every result ZIP contains the source image.
        """
        src = Path(image_path)
        if not src.is_file():
            logger.warning("Source image not found at %s — ZIP will omit uploaded_input", image_path)
            return list(output_files)

        ext = src.suffix.lower() if src.suffix else ".dat"
        dest = out_dir / f"uploaded_input{ext}"
        shutil.copy2(src, dest)
        dest_str = str(dest.resolve())

        merged: list[str] = [dest_str]
        for fp in output_files:
            try:
                if Path(fp).resolve() == dest.resolve():
                    continue
            except OSError:
                pass
            merged.append(fp)
        return merged

    # ------------------------------------------------------------------
    # Mapping helper
    # ------------------------------------------------------------------

    @staticmethod
    def _to_response(entity: JobEntity) -> JobResponse:
        return JobResponse(
            job_id=entity.id,
            model_name=entity.model_name,
            status=entity.status,
            progress=round(entity.progress, 2),
            message=entity.message,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            output_files=[Path(f).name for f in entity.output_files],
            zip_ready=entity.zip_path is not None,
            error=entity.error,
        )
