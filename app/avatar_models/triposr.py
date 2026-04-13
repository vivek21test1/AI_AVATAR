"""
TripoSR model wrapper.

HF repo : stabilityai/TripoSR
VRAM    : 6 GB   |  License: MIT  |  Commercial: Yes
Output  : Textured 3D mesh (.obj) in ~0.5 s

The `tsr` package must be installed:
    pip install git+https://github.com/VAST-AI-Research/TripoSR.git

Weights are loaded from local_dir (model_cache/triposr/) via
TSR.from_pretrained so no network call is made at runtime.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from app.avatar_models.base import BaseAvatarModel


class TripoSRModel(BaseAvatarModel):

    def __init__(self, local_dir: str, repo_dir: str, device: str = "auto") -> None:
        super().__init__("triposr", local_dir, device)
        self.repo_dir = Path(repo_dir)
        self._rembg_session = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self.repo_dir and str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        from tsr.system import TSR  # noqa: PLC0415

        self._model = TSR.from_pretrained(
            str(self.local_dir),
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self._model.renderer.set_chunk_size(8192)
        self._model.to(self.device)

        try:
            import rembg
            self._rembg_session = rembg.new_session()
        except ImportError:
            self.logger.warning("rembg not installed — background removal skipped")

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, image_path: str, output_dir: str, **kwargs) -> dict:
        import torch

        self.ensure_loaded()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path).convert("RGBA")

        if self._rembg_session is not None:
            import rembg
            image = rembg.remove(image, session=self._rembg_session)

        image = self._resize_foreground(image, ratio=0.85)

        # Composite onto white background
        bg = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "RGBA":
            bg.paste(image, mask=image.split()[3])
        else:
            bg.paste(image)

        resolution: int = kwargs.get("resolution", 256)

        with torch.no_grad():
            scene_codes = self._model([bg], device=self.device)

        meshes = self._model.extract_mesh(scene_codes, resolution=resolution)

        obj_path = str(out / "mesh.obj")
        meshes[0].export(obj_path)

        return {
            "output_files": [str(p) for p in out.iterdir() if p.is_file()],
            "output_format": "obj",
            "metadata": {"resolution": resolution},
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_foreground(image: Image.Image, ratio: float) -> Image.Image:
        arr = np.array(image)
        if arr.ndim < 3 or arr.shape[2] < 4:
            return image

        alpha = arr[:, :, 3]
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)
        if not rows.any():
            return image

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        fg_h = rmax - rmin + 1
        fg_w = cmax - cmin + 1
        target = int(min(image.width, image.height) * ratio)
        scale  = target / max(fg_h, fg_w)
        new_w, new_h = int(fg_w * scale), int(fg_h * scale)

        fg = image.crop((cmin, rmin, cmax + 1, rmax + 1)).resize(
            (new_w, new_h), Image.LANCZOS
        )
        result = Image.new("RGBA", image.size, (0, 0, 0, 0))
        px, py = (image.width - new_w) // 2, (image.height - new_h) // 2
        mask = fg.split()[3] if fg.mode == "RGBA" else None
        result.paste(fg, (px, py), mask=mask)
        return result
