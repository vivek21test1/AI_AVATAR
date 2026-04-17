"""
InstantMesh model wrapper.

HF repo : TencentARC/InstantMesh
VRAM    : 10–16 GB  |  License: Apache-2.0  |  Commercial: Yes
Output  : High-quality textured 3D mesh (.obj + texture maps)
Repo    : https://github.com/TencentARC/InstantMesh  (cloned to repos/InstantMesh/)

Pipeline:
  Stage 1 – Zero123++ generates 6 multi-view images from the input.
  Stage 2 – InstantMesh LRM transformer + FlexiCubes reconstructs the mesh.

Directory layout inside model_cache/instantmesh/ varies across HF repo
revisions.  _find_mv_pipeline_dir() probes several strategies so the code
works regardless of whether the sub-directory is named "zero123plus",
"zero123plus-v1.1", or anything else.
"""

import sys
from pathlib import Path
from typing import List, Optional

from PIL import Image

from app.avatar_models.base import BaseAvatarModel


class InstantMeshModel(BaseAvatarModel):

    def __init__(self, local_dir: str, repo_dir: str, device: str = "auto") -> None:
        super().__init__("instantmesh", local_dir, device)
        self.repo_dir = Path(repo_dir)
        self._mv_pipeline = None   # Zero123++ multi-view stage
        self._mesh_model   = None  # InstantMesh reconstruction stage

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        import torch
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
        from omegaconf import OmegaConf

        repo_str = str(self.repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        # ── Stage 1: Zero123++ multi-view pipeline ────────────────────
        mv_weights = self._find_mv_pipeline_dir()
        self.logger.info(f"Loading Zero123++ pipeline from: {mv_weights}")

        self._mv_pipeline = DiffusionPipeline.from_pretrained(
            str(mv_weights),
            custom_pipeline=str(mv_weights),
            torch_dtype=torch.float16,
        )
        self._mv_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self._mv_pipeline.scheduler.config, timestep_spacing="trailing"
        )
        self._mv_pipeline.to(self.device)

        # ── Stage 2: InstantMesh reconstruction model ─────────────────
        cfg_candidates = list(self.repo_dir.glob("configs/*.yaml"))
        cfg_path = next(
            (c for c in cfg_candidates if "large" in c.name),
            cfg_candidates[0] if cfg_candidates else None,
        )
        if cfg_path:
            try:
                from src.utils.train_util import instantiate_from_config  # noqa: PLC0415

                cfg = OmegaConf.load(cfg_path)
                self._mesh_model = instantiate_from_config(cfg.model_config)

                ckpt = (
                    next(Path(self.local_dir).glob("**/*.ckpt"), None)
                    or next(Path(self.local_dir).glob("**/*.safetensors"), None)
                )
                if ckpt:
                    if str(ckpt).endswith(".safetensors"):
                        import safetensors.torch as st
                        state = st.load_file(str(ckpt))
                    else:
                        state = torch.load(str(ckpt), map_location="cpu")
                    self._mesh_model.load_state_dict(state, strict=False)

                self._mesh_model = self._mesh_model.to(self.device).eval()
                self.logger.info("InstantMesh reconstruction model loaded.")
            except Exception as exc:
                self.logger.warning(
                    f"Could not load reconstruction model ({exc}); "
                    "will return multi-view images only."
                )
                self._mesh_model = None
        else:
            self.logger.warning(
                "No YAML config found in repos/InstantMesh/configs/ — "
                "reconstruction model skipped; multi-view output only."
            )

        self._model = self._mv_pipeline

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, image_path: str, output_dir: str, **kwargs) -> dict:
        import torch

        self.ensure_loaded()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        image        = Image.open(image_path).convert("RGB")
        seed: int    = kwargs.get("seed", 42)
        mv_steps: int = kwargs.get("mv_steps", 75)
        use_texmap: bool = kwargs.get("export_texmap", True)

        # ── Stage 1: multi-view generation ───────────────────────────
        generator = torch.Generator(device=self.device).manual_seed(seed)
        mv_grid: Image.Image = self._mv_pipeline(
            image, num_inference_steps=mv_steps, generator=generator
        ).images[0]

        mv_path = str(out / "multiview.png")
        mv_grid.save(mv_path)

        # ── Stage 2: 3-D reconstruction ───────────────────────────────
        if self._mesh_model is not None:
            try:
                output_files = self._reconstruct(mv_grid, out, use_texmap)
            except Exception as exc:
                self.logger.warning(f"Reconstruction failed: {exc} — returning multi-view only")
                output_files = [mv_path]
        else:
            self.logger.warning("Mesh model not loaded — returning multi-view only")
            output_files = [mv_path]

        return {
            "output_files": output_files,
            "output_format": "obj",
            "metadata": {"mv_steps": mv_steps, "export_texmap": use_texmap},
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_mv_pipeline_dir(self) -> Path:
        """
        Locate the Zero123++ diffusion pipeline directory inside the downloaded
        InstantMesh weights.  The sub-directory name varies across HF repo
        revisions.  Strategies tried in order:

          1. Exact 'zero123plus/' sub-directory containing model_index.json
          2. Any sub-directory matching 'zero123plus*' with model_index.json
          3. Any sub-directory (one level deep) that contains model_index.json
          4. model_cache/instantmesh/ itself (last resort — will fail with a
             clear OSError from diffusers if model_index.json is missing)
        """
        base = Path(self.local_dir)

        # Strategy 1: exact canonical name
        exact = base / "zero123plus"
        if exact.is_dir() and (exact / "model_index.json").exists():
            return exact

        # Strategy 2: zero123plus* glob (handles zero123plus-v1.1, zero123plus-v2 …)
        for candidate in sorted(base.glob("zero123plus*")):
            if candidate.is_dir() and (candidate / "model_index.json").exists():
                self.logger.info(
                    f"Found Zero123++ pipeline in '{candidate.name}' "
                    f"(not the expected 'zero123plus')"
                )
                return candidate

        # Strategy 3: any immediate sub-directory with model_index.json
        for sub in sorted(base.iterdir()):
            if sub.is_dir() and (sub / "model_index.json").exists():
                self.logger.info(
                    f"Found diffusion pipeline in sub-directory '{sub.name}'"
                )
                return sub

        # Strategy 4: root itself (let diffusers raise a meaningful error)
        self.logger.warning(
            f"No sub-directory with model_index.json found under {base}. "
            "Falling back to the root weights directory — this will likely "
            "fail.  Check the actual layout with: ls -la /data/model_cache/instantmesh/"
        )
        return base

    def _reconstruct(self, mv_grid: Image.Image, out: Path, use_texmap: bool) -> List[str]:
        import torch
        from torchvision.transforms import v2

        tile_w = mv_grid.width  // 3
        tile_h = mv_grid.height // 2
        views = [
            mv_grid.crop((c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h))
            for r in range(2) for c in range(3)
        ]

        transform = v2.Compose([
            v2.Resize((320, 320)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5]),
        ])
        imgs = torch.stack([transform(v) for v in views]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            planes = self._mesh_model.forward_planes(imgs)
            mesh_v, mesh_f, tex = self._mesh_model.extract_mesh(
                planes, use_texture_map=use_texmap
            )

        from src.utils.mesh_util import save_obj, save_obj_with_mtl  # noqa: PLC0415

        obj_path = str(out / "mesh.obj")
        if use_texmap and tex is not None:
            save_obj_with_mtl(mesh_v, mesh_f, tex, out / "mesh")
        else:
            save_obj(mesh_v, mesh_f, obj_path)

        return [str(p) for p in out.iterdir() if p.is_file()]
