"""
Job repository — thread-safe in-memory store for Job entities.

Job state is also persisted to ``OUTPUT_DIR/{job_id}/job_state.json`` so
that completed results survive a service restart.  On startup every saved
state file is scanned and loaded; jobs that were interrupted mid-run are
automatically marked FAILED.
"""

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from app.core.config import settings
from app.schemas.job_schema import JobStatus

logger = logging.getLogger(__name__)


@dataclass
class JobEntity:
    """Internal mutable job state — not exposed directly to the API."""
    id: str
    model_name: str
    image_path: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    progress: float          = 0.0
    message: str             = ""
    output_files: list       = field(default_factory=list)
    zip_path: Optional[str]  = None
    error: Optional[str]     = None


class JobRepository:

    def __init__(self) -> None:
        self._store: dict = {}   # job_id -> JobEntity
        self._lock = threading.Lock()
        self._load_persisted_jobs()   # restore completed/failed jobs from disk

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, model_name: str, image_path: str) -> JobEntity:
        now = datetime.now(timezone.utc)
        entity = JobEntity(
            id=str(uuid.uuid4()),
            model_name=model_name,
            image_path=image_path,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._store[entity.id] = entity
        return entity

    def get(self, job_id: str) -> Optional[JobEntity]:
        return self._store.get(job_id)

    def list_all(self) -> List[JobEntity]:
        with self._lock:
            return list(self._store.values())

    def update(self, job_id: str, **fields) -> Optional[JobEntity]:
        """Atomically update arbitrary fields on a job entity."""
        with self._lock:
            entity = self._store.get(job_id)
            if entity is None:
                return None
            for key, value in fields.items():
                setattr(entity, key, value)
            entity.updated_at = datetime.now(timezone.utc)

        # Persist to disk when the job reaches a terminal state
        if entity.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            self._save(entity)
        return entity

    def delete(self, job_id: str) -> Optional[JobEntity]:
        """Remove entity from memory; caller handles file cleanup."""
        with self._lock:
            entity = self._store.pop(job_id, None)
        if entity is not None:
            self._delete_state_file(job_id)
        return entity

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _state_file(self, job_id: str) -> Path:
        return settings.OUTPUT_DIR / job_id / "job_state.json"

    def _save(self, entity: JobEntity) -> None:
        """Write job state to disk (non-fatal if it fails)."""
        try:
            state_file = self._state_file(entity.id)
            state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "id": entity.id,
                "model_name": entity.model_name,
                "image_path": entity.image_path,
                "status": entity.status.value,
                "created_at": entity.created_at.isoformat(),
                "updated_at": entity.updated_at.isoformat(),
                "progress": entity.progress,
                "message": entity.message,
                "output_files": entity.output_files,
                "zip_path": entity.zip_path,
                "error": entity.error,
            }
            state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.debug("Persisted job state for %s → %s", entity.id, state_file)
        except Exception as exc:
            logger.warning("Could not persist job state for %s: %s", entity.id, exc)

    def _delete_state_file(self, job_id: str) -> None:
        try:
            self._state_file(job_id).unlink(missing_ok=True)
        except Exception:
            pass

    def _load_persisted_jobs(self) -> None:
        """
        Scan OUTPUT_DIR for ``*/job_state.json`` files and restore job
        entities from them.  Called once at process startup.

        Rules:
        - COMPLETED jobs whose ZIP still exists on disk → restored as-is.
        - COMPLETED jobs whose ZIP is missing            → restored as FAILED.
        - Any other status (PENDING / DOWNLOADING / PROCESSING)
          means the service crashed mid-run               → restored as FAILED.
        """
        try:
            if not settings.OUTPUT_DIR.exists():
                return
        except Exception:
            return

        for state_file in settings.OUTPUT_DIR.glob("*/job_state.json"):
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))

                status = JobStatus(data["status"])
                zip_path: Optional[str] = data.get("zip_path")
                error: Optional[str] = data.get("error")
                message: str = data.get("message", "")

                # Jobs that were interrupted before finishing
                if status not in (JobStatus.COMPLETED, JobStatus.FAILED):
                    status = JobStatus.FAILED
                    error = "Service was restarted before this job could finish."
                    message = "Interrupted by service restart."
                    zip_path = None

                # Completed jobs whose result file was deleted externally
                elif status == JobStatus.COMPLETED:
                    if not zip_path or not Path(zip_path).exists():
                        status = JobStatus.FAILED
                        error = "Result ZIP no longer exists on disk."
                        message = "Result file missing after restart."
                        zip_path = None

                entity = JobEntity(
                    id=data["id"],
                    model_name=data["model_name"],
                    image_path=data.get("image_path", ""),
                    status=status,
                    created_at=datetime.fromisoformat(data["created_at"]),
                    updated_at=datetime.fromisoformat(data["updated_at"]),
                    progress=data.get("progress", 0.0),
                    message=message,
                    output_files=data.get("output_files", []),
                    zip_path=zip_path,
                    error=error,
                )
                self._store[entity.id] = entity
                logger.info(
                    "Restored job %s (model=%s, status=%s) from %s",
                    entity.id, entity.model_name, entity.status.value, state_file,
                )
            except Exception as exc:
                logger.warning("Could not restore job from %s: %s", state_file, exc)
