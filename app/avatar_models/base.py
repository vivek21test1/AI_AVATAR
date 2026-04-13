"""
BaseAvatarModel — abstract contract every ML model wrapper must satisfy.

Subclasses implement:
  load()      – load weights into GPU/CPU memory
  generate()  – run inference on one image, return a result dict
"""

import gc
import logging
from abc import ABC, abstractmethod
from pathlib import Path


class BaseAvatarModel(ABC):

    def __init__(self, model_name: str, local_dir: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.local_dir  = Path(local_dir)
        self.device     = self._resolve_device(device)
        self._model     = None
        self._loaded    = False
        self.logger     = logging.getLogger(f"avatar_models.{model_name}")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory.  Called lazily on first generate()."""

    @abstractmethod
    def generate(self, image_path: str, output_dir: str, **kwargs) -> dict:
        """
        Run inference on a single image.

        Parameters
        ----------
        image_path : str   – path to the input image
        output_dir : str   – directory to write output files into
        **kwargs           – model-specific overrides (resolution, steps, …)

        Returns
        -------
        dict:
          output_files : list[str]  – absolute paths to all generated files
          output_format : str       – "obj", "png", or "ply"
          metadata : dict           – optional extra info
        """

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.logger.info(f"Loading on {self.device} …")
            self.load()
            self._loaded = True
            self.logger.info("Ready")

    def unload(self) -> None:
        """Release GPU/CPU memory."""
        self._model  = None
        self._loaded = False
        gc.collect()
        try:
            import torch
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except ImportError:
            pass
        self.logger.info("Unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
