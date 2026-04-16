"""
Application settings — single source of truth for paths and limits.
Import `settings` wherever configuration values are needed.

Data directory can be overridden via the AI_AVATAR_DATA_DIR environment variable:
    export AI_AVATAR_DATA_DIR=/path/to/data
"""

import os
from pathlib import Path


class Settings:
    # Resolved relative to this file: app/core/ → app/ → project root
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    # Override with AI_AVATAR_DATA_DIR env var (e.g. export AI_AVATAR_DATA_DIR=./data)
    DATA_DIR: Path = Path(os.environ.get("AI_AVATAR_DATA_DIR", "/data"))

    MODEL_CACHE_DIR: Path = DATA_DIR / "model_cache"
    REPOS_DIR: Path       = BASE_DIR / "repos"
    OUTPUT_DIR: Path      = DATA_DIR / "outputs"

    MAX_IMAGE_SIZE_MB: int        = 50
    MAX_JOB_TIMEOUT_SECONDS: int  = 1800   # 30 min

    SUPPORTED_IMAGE_EXTENSIONS: frozenset = frozenset(
        {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )

    def __init__(self) -> None:
        for directory in (self.MODEL_CACHE_DIR, self.REPOS_DIR, self.OUTPUT_DIR):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()


