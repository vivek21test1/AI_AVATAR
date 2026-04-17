"""
InstantMesh model wrapper.

HF repo : TencentARC/InstantMesh
VRAM    : 10–16 GB  |  License: Apache-2.0  |  Commercial: Yes
Output  : High-quality textured 3D mesh (.obj + texture maps)
Repo    : https://github.com/TencentARC/InstantMesh  (cloned to repos/InstantMesh/)

Pipeline:
  Stage 1 – Zero123++ v1.1 generates 6 multi-view images from the input.
  Stage 2 – InstantMesh LRM transformer + FlexiCubes reconstructs the mesh.

Note on weights layout
----------------------
TencentARC/InstantMesh on HuggingFace contains ONLY the mesh reconstruction
checkpoints (instant-mesh-large.ckpt / instant-mesh-base.ckpt).  It does NOT
include the Zero123++ diffusion pipeline.

The Zero123++ weights are sourced in this priority order:
  1. model_cache/zero123plus/        (reuse standalone zero123plus download)
  2. model_cache/instantmesh/<dir>/  (any sub-directory with model_index.json)
  3. Downloaded on-the-fly from sudo-ai/zero123plus-v1.1

The custom pipeline code (pipeline_zero123plus.py) is taken from:
  repos/InstantMesh/zero123plus/  when that directory exists (preferred),
  otherwise from whichever weights directory is chosen above.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

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
        mv_weights, pipeline_code = self._resolve_zero123plus()
        self.logger.info(
            "Zero123++ weights : %s", mv_weights
        )
        self.logger.info(
            "Zero123++ pipeline: %s", pipeline_code
        )

        self._mv_pipeline = DiffusionPipeline.from_pretrained(
            str(mv_weights),
            custom_pipeline=str(pipeline_code),
            torch_dtype=torch.float16,
        )
        self._mv_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self._mv_pipeline.scheduler.config, timestep_spacing="trailing"
        )
        self._mv_pipeline.to(self.device)
        self.logger.info("Zero123++ pipeline loaded.")

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
                    self.logger.info("Loading InstantMesh checkpoint: %s", ckpt)
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
                    "Could not load reconstruction model (%s) — "
                    "will return multi-view images only.", exc
                )
                self._mesh_model = None
        else:
            self.logger.warning(
                "No YAML config found in repos/InstantMesh/configs/ — "
                "reconstruction skipped; multi-view output only."
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

        image         = Image.open(image_path).convert("RGB")
        seed: int     = kwargs.get("seed", 42)
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
                self.logger.warning("Reconstruction failed: %s — returning multi-view only", exc)
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
    # Zero123++ resolution
    # ------------------------------------------------------------------

    def _resolve_zero123plus(self) -> Tuple[Path, Path]:
        """
        Return (weights_dir, pipeline_code_dir) for the Zero123++ stage.

        Weights search order
        --------------------
        1. model_cache/zero123plus/           — reuse standalone download
        2. Any sub-dir of model_cache/instantmesh/ that has model_index.json
        3. Download sudo-ai/zero123plus-v1.1 into model_cache/instantmesh/zero123plus-v1.1/

        Pipeline code search order
        --------------------------
        A. repos/InstantMesh/zero123plus/  (InstantMesh's version, preferred)
        B. Same directory as weights       (HF snapshot includes pipeline_zero123plus.py)
        """
        from app.core.config import settings

        # ── locate weights ──────────────────────────────────────────
        weights_dir: Path = self._find_weights(settings)

        # ── locate pipeline code ────────────────────────────────────
        repo_pipeline = self.repo_dir / "zero123plus"
        if repo_pipeline.is_dir() and (repo_pipeline / "pipeline_zero123plus.py").exists():
            pipeline_code = repo_pipeline
        elif (weights_dir / "pipeline_zero123plus.py").exists():
            pipeline_code = weights_dir
        else:
            # Fallback: diffusers will search the weights dir
            pipeline_code = weights_dir

        return weights_dir, pipeline_code

    def _find_weights(self, settings) -> Path:
        """Locate or download Zero123++ weights; return a valid weights Path."""

        # Priority 1: reuse standalone zero123plus model if already downloaded
        standalone = settings.MODEL_CACHE_DIR / "zero123plus"
        if (standalone / "model_index.json").exists():
            self.logger.info(
                "Reusing model_cache/zero123plus/ for InstantMesh multi-view stage."
            )
            return standalone

        # Priority 2: any sub-directory inside our own cache
        base = Path(self.local_dir)
        if base.is_dir():
            for sub in sorted(base.iterdir()):
                if sub.is_dir() and (sub / "model_index.json").exists():
                    self.logger.info(
                        "Found Zero123++ weights in model_cache/instantmesh/%s/", sub.name
                    )
                    return sub

        # Priority 3: download zero123plus-v1.1 (the version InstantMesh was trained with)
        target = base / "zero123plus-v1.1"
        self._download_zero123plus(target)
        return target

    def _download_zero123plus(self, target: Path) -> None:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
        self.logger.info(
            "Zero123++ weights not found — downloading sudo-ai/zero123plus-v1.1 to %s …", target
        )
        target.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id="sudo-ai/zero123plus-v1.1",
                local_dir=str(target),
                local_dir_use_symlinks=False,
                token=token,
            )
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                "Could not download sudo-ai/zero123plus-v1.1 from HuggingFace.\n"
                "Either download the zero123plus model first via POST /api/v1/models/zero123plus/download,\n"
                "or set HF_TOKEN if the repo requires authentication."
            ) from exc

    # ------------------------------------------------------------------
    # Reconstruction helper
    # ------------------------------------------------------------------

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
