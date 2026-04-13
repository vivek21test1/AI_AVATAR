"""
ModelDownloadService — all download-related business logic.

Responsibilities:
  • Download model weights from HuggingFace via snapshot_download
  • Clone GitHub repos that carry inference source code
  • Track which downloads are currently in-flight (thread-safe)
  • Expose a fire-and-forget background download API
"""

import logging
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

from huggingface_hub import snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

from app.core.exceptions import ModelDownloadError, ModelNotFoundError
from app.core.registry import MODEL_REGISTRY
from app.repositories.model_repository import ModelRepository

logger = logging.getLogger(__name__)


class ModelDownloadService:

    def __init__(self, model_repo: ModelRepository) -> None:
        self._model_repo = model_repo
        self._in_flight: set[str] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_downloading(self, model_name: str) -> bool:
        with self._lock:
            return model_name in self._in_flight

    def download(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Download weights + clone repo for *model_name* (blocking).
        Skips steps that are already complete.
        Returns local_dir path.
        """
        if not self._model_repo.exists(model_name):
            raise ModelNotFoundError(model_name)

        cfg = MODEL_REGISTRY[model_name]
        local_dir: str = cfg["local_dir"]

        self._download_weights(model_name, cfg, local_dir, progress_callback)
        self._clone_repo_if_needed(model_name, cfg, progress_callback)

        return local_dir

    def start_background_download(self, model_name: str) -> bool:
        """
        Kick off a daemon thread to download the model.
        Returns True if a new download was started,
        False if one is already in progress.
        """
        with self._lock:
            if model_name in self._in_flight:
                return False
            self._in_flight.add(model_name)

        thread = threading.Thread(
            target=self._background_worker,
            args=(model_name,),
            daemon=True,
        )
        thread.start()
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download_weights(
        self,
        model_name: str,
        cfg: dict,
        local_dir: str,
        cb: Optional[Callable],
    ) -> None:
        if self._model_repo.is_weights_downloaded(model_name):
            self._notify(cb, f"[{model_name}] Weights already cached at {local_dir}")
            return

        hf_repo = cfg["hf_repo"]
        self._notify(cb, f"[{model_name}] Downloading weights from HuggingFace ({hf_repo}) …")
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        try:
            snapshot_download(
                repo_id=hf_repo,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            )
        except RepositoryNotFoundError as exc:
            raise ModelDownloadError(
                f"HuggingFace repository not found: {hf_repo}. "
                "Verify the repo id in app/core/registry.py."
            ) from exc
        except Exception as exc:
            raise ModelDownloadError(
                f"Failed to download weights for '{model_name}': {exc}"
            ) from exc

        self._notify(cb, f"[{model_name}] Weights downloaded → {local_dir}")

    def _clone_repo_if_needed(
        self,
        model_name: str,
        cfg: dict,
        cb: Optional[Callable],
    ) -> None:
        github_url = cfg.get("github_repo")
        repo_dir   = cfg.get("repo_dir")
        if not github_url or not repo_dir:
            return

        if self._model_repo.is_repo_cloned(model_name):
            self._notify(cb, f"[{model_name}] Repo already cloned at {repo_dir}")
            return

        self._notify(cb, f"[{model_name}] Cloning {github_url} …")
        Path(repo_dir).parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth=1", github_url, repo_dir],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ModelDownloadError(
                f"git clone failed for {github_url}:\n{result.stderr}"
            )
        self._notify(cb, f"[{model_name}] Repo cloned → {repo_dir}")

    def _background_worker(self, model_name: str) -> None:
        try:
            self.download(model_name)
            logger.info(f"Background download complete: '{model_name}'")
        except Exception as exc:
            logger.error(f"Background download failed for '{model_name}': {exc}")
        finally:
            with self._lock:
                self._in_flight.discard(model_name)

    @staticmethod
    def _notify(cb: Optional[Callable], msg: str) -> None:
        logger.info(msg)
        if cb:
            cb(msg)
