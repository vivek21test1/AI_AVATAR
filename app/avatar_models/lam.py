"""
LAM (Large Avatar Model) wrapper.

HF repo : aigc3d/LAM
VRAM    : ~12–16 GB  |  License: Apache-2.0  |  Commercial: Yes
Output  : Animatable 3D Gaussian Splatting head avatar (.ply + metadata)
Repo    : https://github.com/aigc3d/LAM  (SIGGRAPH 2025)

Takes a single portrait image and produces a driveable 3D head represented
as 3D Gaussian splats stored in a .ply file.  Open the output in a 3DGS-
compatible viewer such as SuperSplat or KIRI Engine.
"""

import sys
from pathlib import Path

from PIL import Image

from app.avatar_models.base import BaseAvatarModel


class LAMModel(BaseAvatarModel):

    def __init__(self, local_dir: str, repo_dir: str, device: str = "auto") -> None:
        super().__init__("lam", local_dir, device)
        self.repo_dir = Path(repo_dir)
        self._engine = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        repo_str = str(self.repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        from lam.runner.infer_lam import LAMInferencer  # noqa: PLC0415

        ckpt = (
            next(Path(self.local_dir).glob("**/*.pth"), None)
            or next(Path(self.local_dir).glob("**/*.ckpt"), None)
            or next(Path(self.local_dir).glob("**/*.safetensors"), None)
        )
        if ckpt is None:
            raise RuntimeError(
                f"No checkpoint found in {self.local_dir}. "
                "Ensure the model is fully downloaded."
            )

        self._engine = LAMInferencer(model_path=str(ckpt), device=self.device)
        self._model  = self._engine

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, image_path: str, output_dir: str, **kwargs) -> dict:
        self.ensure_loaded()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # LAM expects 512×512 portrait input
        image = Image.open(image_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        preprocessed = str(out / "input_512.png")
        image.save(preprocessed)

        ply_path      = str(out / "avatar.ply")
        metadata_path = str(out / "avatar_metadata.json")

        self._engine.infer(
            image_path=preprocessed,
            output_ply_path=ply_path,
            output_metadata_path=metadata_path,
        )

        output_files = [
            f for f in (ply_path, metadata_path, preprocessed)
            if Path(f).exists()
        ]

        return {
            "output_files": output_files,
            "output_format": "ply",
            "metadata": {
                "description": "3D Gaussian Splatting animatable head avatar",
                "viewer": "SuperSplat / KIRI Engine (open avatar.ply)",
            },
        }
