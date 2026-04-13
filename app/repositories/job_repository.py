"""
Job repository — thread-safe in-memory store for Job entities.

Separates persistence concerns from business logic (JobService).
JobEntity is the internal mutable state object; JobResponse (schema) is
built by the service when returning data to the API layer.
"""

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from app.schemas.job_schema import JobStatus


@dataclass
class JobEntity:
    """Internal mutable job state — not exposed directly to the API."""
    id: str
    model_name: str
    image_path: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    progress: float       = 0.0
    message: str          = ""
    output_files: list    = field(default_factory=list)
    zip_path: Optional[str] = None
    error: Optional[str]    = None


class JobRepository:

    def __init__(self) -> None:
        self._store: dict[str, JobEntity] = {}
        self._lock = threading.Lock()

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

    def list_all(self) -> list[JobEntity]:
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
        return entity

    def delete(self, job_id: str) -> Optional[JobEntity]:
        """Remove and return the entity (caller is responsible for file cleanup)."""
        with self._lock:
            return self._store.pop(job_id, None)
