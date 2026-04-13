"""
Model repository — read-only access to the model registry plus
local-filesystem checks for whether weights / repos are present.

This layer is the only place that reads MODEL_REGISTRY and touches the file
system to verify download state.  Services call this instead of doing their
own Path checks.
"""

from pathlib import Path
from typing import Optional

from app.core.registry import MODEL_REGISTRY

_WEIGHT_EXTENSIONS = {".safetensors", ".bin", ".ckpt", ".pth", ".pt"}


class ModelRepository:

    # ------------------------------------------------------------------
    # Registry queries
    # ------------------------------------------------------------------

    def get_all_names(self) -> list[str]:
        return list(MODEL_REGISTRY.keys())

    def get_config(self, model_name: str) -> Optional[dict]:
        """Return the registry entry for *model_name*, or None if unknown."""
        return MODEL_REGISTRY.get(model_name)

    def exists(self, model_name: str) -> bool:
        return model_name in MODEL_REGISTRY

    # ------------------------------------------------------------------
    # Local-filesystem checks
    # ------------------------------------------------------------------

    def is_weights_downloaded(self, model_name: str) -> bool:
        """
        True when model weights exist locally.
        Uses key_files list if specified; falls back to weight-extension scan.
        """
        cfg = MODEL_REGISTRY.get(model_name, {})
        local_dir = Path(cfg.get("local_dir", ""))
        key_files: list[str] = cfg.get("key_files", [])

        if not local_dir.exists():
            return False

        if key_files:
            return all((local_dir / f).exists() for f in key_files)

        # Fallback: any recognised weight file anywhere under local_dir
        return any(
            f.suffix in _WEIGHT_EXTENSIONS
            for f in local_dir.rglob("*")
            if f.is_file()
        )

    def is_repo_cloned(self, model_name: str) -> bool:
        """True when the GitHub inference-code repo is already cloned."""
        cfg = MODEL_REGISTRY.get(model_name, {})
        repo_dir = cfg.get("repo_dir")
        if not repo_dir:
            return True   # no repo needed for this model
        return (Path(repo_dir) / ".git").exists()

    def is_ready(self, model_name: str) -> bool:
        """True when both weights and repo (if any) are present."""
        return self.is_weights_downloaded(model_name) and self.is_repo_cloned(model_name)
