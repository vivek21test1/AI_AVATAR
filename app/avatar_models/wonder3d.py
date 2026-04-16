"""
Wonder3D model wrapper.

HF repo : flamehaze1115/wonder3d-v1.0
VRAM    : 8 GB   |  License: MIT  |  Commercial: Yes
Output  : Multi-view normal maps + colour images → textured 3D mesh
Repo    : https://github.com/xxlong0/Wonder3D  (cloned to repos/Wonder3D/)

Uses a cross-domain diffusion model to generate 6 consistent normal + colour
views, then optionally reconstructs a mesh via NeuS (instant-nsr-pl).
"""

import sys
from pathlib import Path
from typing import List, Optional

from PIL import Image

from app.avatar_models.base import BaseAvatarModel


class Wonder3DModel(BaseAvatarModel):

    def __init__(self, local_dir: str, repo_dir: str, device: str = "auto") -> None:
        super().__init__("wonder3d", local_dir, device)
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

        from mvdiffusion.pipelines.pipeline_mvdiffusion_image import (  # noqa: PLC0415
            MVDiffusionImagePipeline,
        )

        cfg_path = self.repo_dir / "configs" / "mvdiffusion-joint-ortho-6views.yaml"
        # Convert the whole config to a plain dict first, then extract the key.
        # Using .get() on an OmegaConf node returns a plain {} when the key is
        # absent, which OmegaConf.to_container() rejects as a non-OmegaConf object.
        cfg_dict = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        extra_kwargs = cfg_dict.get("pipeline_kwargs", {})

        self._pipeline = MVDiffusionImagePipeline.from_pretrained(
            str(self.local_dir),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            **extra_kwargs,
        )
        self._pipeline.to(self.device)
        self._model = self._pipeline

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, image_path: str, output_dir: str, **kwargs) -> dict:
        import torch

        self.ensure_loaded()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path).convert("RGB").resize((256, 256))

        steps: float   = kwargs.get("num_inference_steps", 50)
        guidance: float = kwargs.get("guidance_scale", 2.0)
        seed: int       = kwargs.get("seed", 42)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self._pipeline(
            image=image,          # single PIL image — pipeline replicates to num_views*2 internally
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            output_type="pil",
        )

        output_files: List[str] = []
        for i, img in enumerate(result.images):
            p = str(out / f"view_{i:02d}.png")
            img.save(p)
            output_files.append(p)

        # Optional mesh reconstruction (requires instant-nsr-pl in the repo)
        mesh_path = self._try_reconstruct_mesh(output_files, out)
        if mesh_path:
            output_files.append(mesh_path)

        return {
            "output_files": output_files,
            "output_format": "obj",
            "metadata": {"num_inference_steps": steps, "views": len(result.images)},
        }

    # ------------------------------------------------------------------
    # Mesh reconstruction
    # ------------------------------------------------------------------

    def _try_reconstruct_mesh(self, view_paths: List[str], out: Path) -> Optional[str]:
        try:
            from instant_nsr_pl.run import reconstruct_from_views  # noqa: PLC0415

            obj_path = str(out / "mesh.obj")
            reconstruct_from_views(view_paths=view_paths, output_path=obj_path)
            return obj_path
        except ImportError:
            self.logger.warning("instant_nsr_pl not found — returning multi-view images only")
            return None
        except Exception as exc:
            self.logger.warning(f"Mesh reconstruction failed: {exc}")
            return None
