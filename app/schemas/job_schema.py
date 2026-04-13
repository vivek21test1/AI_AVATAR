from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING     = "pending"
    DOWNLOADING = "downloading"
    PROCESSING  = "processing"
    COMPLETED   = "completed"
    FAILED      = "failed"


class JobResponse(BaseModel):
    job_id: str
    model_name: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    message: str
    created_at: datetime
    updated_at: datetime
    output_files: list[str]      # file names only (no full paths)
    zip_ready: bool
    error: Optional[str] = None


class JobDeleteResponse(BaseModel):
    deleted: bool
    job_id: str
