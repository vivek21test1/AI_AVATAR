"""
InstantMesh model wrapper.

HF repo : TencentARC/InstantMesh
VRAM    : 10–16 GB  |  License: Apache-2.0  |  Commercial: Yes
Output  : High-quality textured 3D mesh (.obj + texture maps)
Repo    : https://github.com/TencentARC/InstantMesh  (cloned to repos/InstantMesh/)

Pipeline:
  Stage 1 – Zero123++ v1.1 generates 6 multi-view images from the input.
  Stage 2 – InstantMesh LRM transformer + FlexiCubes reconstructs the mesh.

Zero123++ weights search order
-------------------------------
1. model_cache/zero123plus/        — reuse standalone zero123plus download
2. model_cache/instantmesh/<dir>/  — any sub-directory with model_index.json
3. Download on-the-fly from sudo-ai/zero123plus-v1.1

pipeline_zero123plus.py search order
--------------------------------------
A. Recursive search under repos/InstantMesh/  (cloned GitHub repo, any sub-path)
B. Recursive search under <weights_dir>/
C. model_cache/zero123plus/pipeline_zero123plus.py  (standalone model cache)
D. HuggingFace hf_hub_download sudo-ai/zero123plus  (needs HF_TOKEN if gated)
E. None  → from_pretrained without custom_pipeline  (last-resort; may fail if
           Zero123PlusPipeline is not in the installed diffusers version)

If E also fails the user must supply pipeline_zero123plus.py manually.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

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

        # ── Ensure inference repo is cloned (needed for configs/ and src/) ──
        if not (self.repo_dir / ".git").exists():
            self._clone_repo()

        repo_str = str(self.repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        # ── Stage 1: Zero123++ multi-view pipeline ────────────────────
        mv_weights: Path = self._find_weights()
        pipeline_py: Optional[str] = self._find_pipeline_py(mv_weights)

        self.logger.info("Zero123++ weights : %s", mv_weights)
        self.logger.info("Zero123++ pipeline: %s", pipeline_py or "(none — using model_index class)")

        pretrained_kwargs = {"torch_dtype": torch.float16}
        if pipeline_py is not None:
            pretrained_kwargs["custom_pipeline"] = pipeline_py

        try:
            self._mv_pipeline = DiffusionPipeline.from_pretrained(
                str(mv_weights), **pretrained_kwargs
            )
        except Exception as exc:
            if pipeline_py is None:
                # Already tried without custom_pipeline — nothing more to do
                raise RuntimeError(
                    "Cannot load Zero123++ pipeline.\n"
                    "pipeline_zero123plus.py could not be found and "
                    "DiffusionPipeline.from_pretrained failed without it.\n\n"
                    "Fix (choose one):\n"
                    "  1. Set HF_TOKEN and accept terms at "
                    "https://huggingface.co/sudo-ai/zero123plus, then retry.\n"
                    "  2. Download the zero123plus model first: "
                    "POST /api/v1/models/zero123plus/download\n"
                    f"  3. Copy pipeline_zero123plus.py manually to {mv_weights}/"
                ) from exc
            raise

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

        generator = torch.Generator(device=self.device).manual_seed(seed)
        mv_grid: Image.Image = self._mv_pipeline(
            image, num_inference_steps=mv_steps, generator=generator
        ).images[0]

        mv_path = str(out / "multiview.png")
        mv_grid.save(mv_path)

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
    # Repo management
    # ------------------------------------------------------------------

    def _clone_repo(self) -> None:
        """Clone TencentARC/InstantMesh to self.repo_dir (shallow, depth=1)."""
        import subprocess

        github_url = "https://github.com/TencentARC/InstantMesh.git"
        self.logger.info("Cloning InstantMesh repo to %s …", self.repo_dir)
        self.repo_dir.parent.mkdir(parents=True, exist_ok=True)

        # Remove a stale partial directory (no .git) before cloning
        if self.repo_dir.exists() and not (self.repo_dir / ".git").exists():
            import shutil
            shutil.rmtree(self.repo_dir, ignore_errors=True)

        result = subprocess.run(
            ["git", "clone", "--depth=1", github_url, str(self.repo_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.logger.warning(
                "git clone failed (returncode=%d): %s — "
                "will continue without repo; configs and src/ utilities unavailable.",
                result.returncode, result.stderr.strip(),
            )
        else:
            self.logger.info("InstantMesh repo cloned → %s", self.repo_dir)

    # ------------------------------------------------------------------
    # Zero123++ helpers
    # ------------------------------------------------------------------

    def _find_weights(self) -> Path:
        """Locate or download Zero123++ weights; return the directory Path."""
        from app.core.config import settings

        # Priority 1: reuse standalone zero123plus model if downloaded
        standalone = settings.MODEL_CACHE_DIR / "zero123plus"
        if (standalone / "model_index.json").exists():
            self.logger.info("Reusing model_cache/zero123plus/ for multi-view stage.")
            return standalone

        # Priority 2: any sub-directory inside InstantMesh's own cache
        base = Path(self.local_dir)
        if base.is_dir():
            for sub in sorted(base.iterdir()):
                if sub.is_dir() and (sub / "model_index.json").exists():
                    self.logger.info(
                        "Found Zero123++ weights in model_cache/instantmesh/%s/", sub.name
                    )
                    return sub

        # Priority 3: download zero123plus-v1.1
        target = base / "zero123plus-v1.1"
        self._download_zero123plus(target)
        return target

    def _find_pipeline_py(self, weights_dir: Path) -> Optional[str]:
        """
        Search all known locations for pipeline_zero123plus.py.
        Returns the absolute path as a string, or None if not found anywhere.
        Never raises — callers handle the None case.
        """
        from app.core.config import settings

        # ── Explicit high-priority candidates ──────────────────────────
        explicit = [
            weights_dir / "pipeline_zero123plus.py",
            settings.MODEL_CACHE_DIR / "zero123plus" / "pipeline_zero123plus.py",
        ]
        for p in explicit:
            if p.is_file():
                self.logger.info("Found pipeline_zero123plus.py at %s", p)
                return str(p)

        # ── Recursive search: entire cloned repo (any sub-path) ────────
        if self.repo_dir.is_dir():
            found = next(self.repo_dir.rglob("pipeline_zero123plus.py"), None)
            if found is not None:
                self.logger.info("Found pipeline_zero123plus.py in repo: %s", found)
                return str(found)

        # ── Recursive search: weights directory ────────────────────────
        found = next(weights_dir.rglob("pipeline_zero123plus.py"), None)
        if found is not None:
            self.logger.info("Found pipeline_zero123plus.py in weights: %s", found)
            return str(found)

        # ── Try downloading from HuggingFace (needs HF_TOKEN if gated) ─
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
        target = weights_dir / "pipeline_zero123plus.py"
        try:
            from huggingface_hub import hf_hub_download

            self.logger.info(
                "pipeline_zero123plus.py not found locally — "
                "attempting HF download (sudo-ai/zero123plus) …"
            )
            hf_hub_download(
                repo_id="sudo-ai/zero123plus",
                filename="pipeline_zero123plus.py",
                local_dir=str(target.parent),
                local_dir_use_symlinks=False,
                token=token,
            )
            if target.is_file():
                self.logger.info("Downloaded pipeline_zero123plus.py → %s", target)
                return str(target)
        except Exception as exc:
            token_hint = (
                " (tip: set HF_TOKEN — the repo may require authentication)"
                if token is None else ""
            )
            self.logger.warning(
                "HF download of pipeline_zero123plus.py failed%s: %s",
                token_hint, exc,
            )

        # ── Nothing worked ─────────────────────────────────────────────
        self.logger.warning(
            "pipeline_zero123plus.py could not be found or downloaded. "
            "Will attempt DiffusionPipeline.from_pretrained without custom_pipeline. "
            "If that fails, fix:\n"
            "  1. Set HF_TOKEN and accept terms at "
            "https://huggingface.co/sudo-ai/zero123plus\n"
            "  2. Or: POST /api/v1/models/zero123plus/download first\n"
            "  3. Or: copy pipeline_zero123plus.py manually to %s",
            target,
        )
        return None

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
                "Either download the zero123plus model first via "
                "POST /api/v1/models/zero123plus/download,\n"
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
