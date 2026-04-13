from pydantic import BaseModel


class GenerateJobResponse(BaseModel):
    """Returned immediately when a generation job is submitted (async path)."""
    job_id: str
    status: str
    model_name: str
    poll_url: str
    result_url: str
