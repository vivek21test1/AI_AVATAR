"""
CRM (Convolutional Reconstruction Model) wrapper.

HF repo : Zhengyi/CRM
VRAM    : 8 GB   |  License: MIT  |  Commercial: Yes
Output  : Full textured 3D mesh (.obj + textures) in ~10 s
Repo    : https://github.com/thu-ml/CRM  (cloned to repos/CRM/)

Pipeline:
  Stage 1 – Zero123++ generates 6 consistent multi-view images.
  Stage 2 – CRM reconstructs a textured mesh from those views.

Inference code (pipelines.py, model.py, …) lives in the cloned repo;
we add it to sys.path at load time.
"""

import sys
from pathlib import Path

from PIL import Image

from app.avatar_models.base import BaseAvatarModel


class CRMModel(BaseAvatarModel):

    def __init__(self, local_dir: str, repo_dir: str, device: str = "auto") -> None:
        super().__init__("crm", local_dir, device)
        self.repo_dir = Path(repo_dir)
        self._pipeline = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        import torch
        from omegaconf import OmegaConf

        repo_str = str(self.repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        from pipelines import TwoStagePipeline  # noqa: PLC0415

        stage1_cfg = OmegaConf.load(self.repo_dir / "configs" / "crm_stage1.yaml")
        stage2_cfg = OmegaConf.load(self.repo_dir / "configs" / "crm_stage2.yaml")

        # Point stage-2 weights at our local cache
        ckpt = next(Path(self.local_dir).glob("**/*.pth"), None)
        if ckpt:
            stage2_cfg.model.resume = str(ckpt)

        self._pipeline = TwoStagePipeline(
            stage1_config=stage1_cfg,
            stage2_config=stage2_cfg,
            device=self.device,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self._model = self._pipeline

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, image_path: str, output_dir: str, **kwargs) -> dict:
        self.ensure_loaded()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        image  = Image.open(image_path).convert("RGBA")
        scale: float = kwargs.get("scale", 5.0)
        steps: int   = kwargs.get("steps", 50)
        seed: int    = kwargs.get("seed", 1)

        result = self._pipeline(image, scale=scale, step=steps, seed=seed)

        mesh = result.get("mesh") or result
        obj_path = str(out / "mesh.obj")

        if hasattr(mesh, "export"):
            mesh.export(obj_path)
        elif isinstance(mesh, (list, tuple)):
            mesh[0].export(obj_path)

        return {
            "output_files": [str(p) for p in out.iterdir() if p.is_file()],
            "output_format": "obj",
            "metadata": {"scale": scale, "steps": steps},
        }
