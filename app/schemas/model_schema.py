from pydantic import BaseModel


class ModelInfoResponse(BaseModel):
    model_name: str
    display_name: str
    hf_repo: str
    vram_gb: int
    output_format: str
    description: str
    license: str
    commercial_use: bool
    weights_downloaded: bool
    repo_cloned: bool
    ready: bool
    download_in_progress: bool


class DownloadTriggerResponse(BaseModel):
    model_name: str
    started: bool
    message: str
