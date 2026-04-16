"""
Generate router — controller for avatar generation endpoints.

POST /generate       – async (returns job_id, poll separately)
POST /generate/sync  – synchronous (blocks, streams ZIP directly)

The router is responsible only for:
  • Reading + validating the uploaded file and form fields
  • Delegating to GenerationService
  • Returning the appropriate HTTP response

No business logic lives here.
"""

import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.exceptions import GenerationError
from app.dependencies import get_generation_service, get_job_service
from app.schemas.generate_schema import GenerateJobResponse
from app.schemas.job_schema import JobStatus
from app.services.generation_service import GenerationService
from app.services.job_service import JobService

router = APIRouter(prefix="/generate", tags=["Generate"])


# ---------------------------------------------------------------------------
# Async generation (recommended)
# ---------------------------------------------------------------------------

@router.post(
    "",
    status_code=202,
    response_model=GenerateJobResponse,
    summary="Submit async generation job",
)
async def generate(
    image: Annotated[UploadFile, File(description="Input image (JPG/PNG/WebP/BMP)")],
    model_name: Annotated[str, Form(description="Model key, e.g. 'triposr'")],
    auto_download: Annotated[bool, Form(description="Auto-download model if not cached")] = True,
    generation_service: GenerationService = Depends(get_generation_service),
):
    """
    Submit a generation job and return a `job_id` immediately.

    Poll `GET /api/v1/jobs/{job_id}` until `status == "completed"`,
    then download the result from `GET /api/v1/jobs/{job_id}/result`.
    """
    image_bytes, extension = await _read_upload(image)

    job = generation_service.submit(
        model_name=model_name,
        image_bytes=image_bytes,
        file_extension=extension,
        auto_download=auto_download,
    )

    return GenerateJobResponse(
        job_id=job.job_id,
        status=job.status.value,
        model_name=model_name,
        poll_url=f"/api/v1/jobs/{job.job_id}",
        result_url=f"/api/v1/jobs/{job.job_id}/result",
    )


# ---------------------------------------------------------------------------
# Synchronous generation (convenience — may time out for slow models)
# ---------------------------------------------------------------------------

@router.post(
    "/sync",
    summary="Synchronous generation (blocks until done)",
)
async def generate_sync(
    image: Annotated[UploadFile, File(description="Input image (JPG/PNG/WebP/BMP)")],
    model_name: Annotated[str, Form(description="Model key, e.g. 'triposr'")],
    auto_download: Annotated[bool, Form(description="Auto-download model if not cached")] = True,
    generation_service: GenerationService = Depends(get_generation_service),
    job_service: JobService = Depends(get_job_service),
):
    """
    Synchronous path — blocks until the avatar is ready, then streams the ZIP.
    Best for fast models (TripoSR ≈ 0.5 s, Zero123++ ≈ 30 s).
    Use the async endpoint for Wonder3D / LAM which can take minutes.
    """
    image_bytes, extension = await _read_upload(image)

    job = generation_service.submit(
        model_name=model_name,
        image_bytes=image_bytes,
        file_extension=extension,
        auto_download=auto_download,
    )

    # Poll until the background thread finishes (max 30 min)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + settings.MAX_JOB_TIMEOUT_SECONDS
    while loop.time() < deadline:
        await asyncio.sleep(2)
        current = job_service.get(job.job_id)
        if current.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            break

    current = job_service.get(job.job_id)

    if current.status == JobStatus.FAILED:
        raise GenerationError(f"Generation failed: {current.error}")

    if current.status != JobStatus.COMPLETED:
        raise HTTPException(504, "Generation timed out — use POST /generate instead.")

    entity = job_service.assert_result_ready(job.job_id)

    return FileResponse(
        path=entity.zip_path,
        media_type="application/zip",
        filename=f"avatar_{model_name}_{job.job_id[:8]}.zip",
    )


# ---------------------------------------------------------------------------
# Shared upload helper (private to this module)
# ---------------------------------------------------------------------------

async def _read_upload(upload: UploadFile) -> tuple[bytes, str]:
    """Validate and read the uploaded image; return (bytes, extension)."""
    ext = Path(upload.filename or "image.png").suffix.lower()

    if ext not in settings.SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported format '{ext}'. "
            f"Allowed: {sorted(settings.SUPPORTED_IMAGE_EXTENSIONS)}",
        )

    data = await upload.read()
    size_mb = len(data) / (1024 * 1024)

    if size_mb > settings.MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            413,
            f"Image too large ({size_mb:.1f} MB). Maximum: {settings.MAX_IMAGE_SIZE_MB} MB.",
        )

    return data, ext
