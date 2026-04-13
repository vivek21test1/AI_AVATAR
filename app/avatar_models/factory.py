"""
ModelFactory — creates and caches avatar model instances.

Design decisions:
  • Only ONE model is kept in GPU memory at a time; loading a second model
    automatically unloads the previous one (VRAM budget management).
  • Model instances are cached so the same object is reused across requests;
    model weights are not reloaded unless the instance was explicitly unloaded.
  • All public methods are thread-safe.
"""

import logging
import threading
from typing import Optional

from app.avatar_models.base import BaseAvatarModel
from app.core.exceptions import ModelNotFoundError
from app.core.registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)


class ModelFactory:
    _cache: dict[str, BaseAvatarModel] = {}
    _lock  = threading.Lock()
    _active: Optional[str] = None    # model currently in VRAM

    @classmethod
    def get(cls, model_name: str, device: str = "auto") -> BaseAvatarModel:
        """
        Return a (possibly cached) model instance ready for inference.
        Raises ModelNotFoundError for unknown names.
        """
        if model_name not in MODEL_REGISTRY:
            raise ModelNotFoundError(model_name)

        with cls._lock:
            if model_name not in cls._cache:
                cls._cache[model_name] = cls._build(model_name, device)

            # Evict the previously active model to free VRAM
            if cls._active and cls._active != model_name:
                prev = cls._cache.get(cls._active)
                if prev and prev.is_loaded:
                    logger.info(
                        f"Evicting '{cls._active}' from VRAM to load '{model_name}'"
                    )
                    prev.unload()

            cls._active = model_name
            return cls._cache[model_name]

    @classmethod
    def evict_all(cls) -> None:
        """Unload every model from GPU memory (useful in low-VRAM scenarios)."""
        with cls._lock:
            for model in cls._cache.values():
                if model.is_loaded:
                    model.unload()
            cls._active = None

    # ------------------------------------------------------------------
    # Internal factory
    # ------------------------------------------------------------------

    @classmethod
    def _build(cls, model_name: str, device: str) -> BaseAvatarModel:
        cfg      = MODEL_REGISTRY[model_name]
        local_dir = cfg["local_dir"]
        repo_dir  = cfg.get("repo_dir", "")

        if model_name == "triposr":
            from app.avatar_models.triposr import TripoSRModel
            return TripoSRModel(local_dir=local_dir, repo_dir=repo_dir, device=device)

        if model_name == "zero123plus":
            from app.avatar_models.zero123plus import Zero123PlusModel
            return Zero123PlusModel(local_dir=local_dir, device=device)

        if model_name == "crm":
            from app.avatar_models.crm import CRMModel
            return CRMModel(local_dir=local_dir, repo_dir=repo_dir, device=device)

        if model_name == "wonder3d":
            from app.avatar_models.wonder3d import Wonder3DModel
            return Wonder3DModel(local_dir=local_dir, repo_dir=repo_dir, device=device)

        if model_name == "instantmesh":
            from app.avatar_models.instantmesh import InstantMeshModel
            return InstantMeshModel(local_dir=local_dir, repo_dir=repo_dir, device=device)

        if model_name == "lam":
            from app.avatar_models.lam import LAMModel
            return LAMModel(local_dir=local_dir, repo_dir=repo_dir, device=device)

        raise ModelNotFoundError(model_name)
