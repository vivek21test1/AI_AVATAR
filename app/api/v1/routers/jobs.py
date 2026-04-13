"""
Jobs router — controller for job lifecycle endpoints.

GET    /jobs            – list all jobs
GET    /jobs/{job_id}   – single job status
GET    /jobs/{job_id}/result – download result ZIP
DELETE /jobs/{job_id}   – delete job + files
"""

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.dependencies import get_job_service
from app.schemas.job_schema import JobDeleteResponse, JobResponse
from app.services.job_service import JobService

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get(
    "",
    response_model=list[JobResponse],
    summary="List all jobs",
)
def list_jobs(job_service: JobService = Depends(get_job_service)):
    """Return all submitted jobs with their current status and progress."""
    return job_service.list_all()


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get job status",
)
def get_job(job_id: str, job_service: JobService = Depends(get_job_service)):
    """Poll this endpoint until `status == "completed"` before downloading."""
    return job_service.get(job_id)


@router.get(
    "/{job_id}/result",
    summary="Download result ZIP",
    response_class=FileResponse,
)
def download_result(job_id: str, job_service: JobService = Depends(get_job_service)):
    """
    Stream the generated avatar as a ZIP file.
    Only available when `status == "completed"`.
    """
    entity = job_service.assert_result_ready(job_id)

    return FileResponse(
        path=entity.zip_path,
        media_type="application/zip",
        filename=f"avatar_{entity.model_name}_{job_id[:8]}.zip",
    )


@router.delete(
    "/{job_id}",
    response_model=JobDeleteResponse,
    summary="Delete job and output files",
)
def delete_job(job_id: str, job_service: JobService = Depends(get_job_service)):
    """Remove the job record and all output files from disk."""
    job_service.delete(job_id)
    return JobDeleteResponse(deleted=True, job_id=job_id)
