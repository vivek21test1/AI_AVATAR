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
        Search all known locations for a file containing Zero123PlusPipeline.
        Returns the absolute path as a string, or None if not found anywhere.
        Never raises — callers handle the None case.
        """
        from app.core.config import settings

        # ── 1. Vendored bundled copy (always present in this project) ───
        vendor_path = Path(__file__).parent.parent / "vendor" / "pipeline_zero123plus.py"
        if vendor_path.is_file():
            self.logger.info("Using vendored pipeline_zero123plus.py: %s", vendor_path)
            return str(vendor_path)

        # ── 2. Exact filename in known locations ────────────────────────
        explicit = [
            weights_dir / "pipeline_zero123plus.py",
            settings.MODEL_CACHE_DIR / "zero123plus" / "pipeline_zero123plus.py",
        ]
        for p in explicit:
            if p.is_file():
                self.logger.info("Found pipeline_zero123plus.py at %s", p)
                return str(p)

        # ── 3. Class-name search in ALL .py files (any filename) ────────
        # Handles renames, subdirectory moves, and alternative copies.
        keyword = "Zero123PlusPipeline"
        for root in [self.repo_dir, weights_dir, settings.MODEL_CACHE_DIR]:
            if not root.is_dir():
                continue
            for py_file in root.rglob("*.py"):
                try:
                    if keyword in py_file.read_text(errors="ignore"):
                        self.logger.info(
                            "Found %s class in %s", keyword, py_file
                        )
                        return str(py_file)
                except Exception:
                    continue

        # ── 4. HuggingFace .py scan (repo may have moved the file) ──────
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
        target = weights_dir / "pipeline_zero123plus.py"
        found_via_hf = self._fetch_pipeline_from_hf(target, token)
        if found_via_hf:
            return str(found_via_hf)

        # ── 5. Clone SUDO-AI-3D/zero123plus GitHub repo and search ──────
        found_via_git = self._fetch_pipeline_from_github(target)
        if found_via_git:
            return str(found_via_git)

        # ── Nothing worked — fall through to no-custom-pipeline attempt ─
        self.logger.warning(
            "All pipeline_zero123plus.py search attempts failed. "
            "Will try DiffusionPipeline.from_pretrained without custom_pipeline."
        )
        return None

    def _fetch_pipeline_from_hf(
        self, dest: Path, token: Optional[str]
    ) -> Optional[Path]:
        """
        Download every ``.py`` file from ``sudo-ai/zero123plus`` and return the
        path of the one that contains the Zero123PlusPipeline class definition,
        saved as *dest*.

        This is necessary because the file location inside the HF repo has
        changed over time and a direct ``hf_hub_download("pipeline_zero123plus.py")``
        returns 404 while the class still exists somewhere in the repo.

        Returns the saved path on success, or None on any failure.
        """
        try:
            from huggingface_hub import snapshot_download

            py_cache = dest.parent / "_hf_py_cache"
            self.logger.info(
                "Scanning sudo-ai/zero123plus for Zero123++ pipeline code "
                "(downloading .py files only) …"
            )
            snapshot_download(
                repo_id="sudo-ai/zero123plus",
                local_dir=str(py_cache),
                local_dir_use_symlinks=False,
                allow_patterns=["*.py"],
                token=token,
            )
        except Exception as exc:
            self.logger.warning(
                "Could not fetch .py files from sudo-ai/zero123plus: %s", exc
            )
            return None

        # Search every .py file for the Zero123Plus pipeline class
        import shutil
        keywords = ("Zero123PlusPipeline", "class Zero123Plus", "zero123plus")
        for py_file in sorted(Path(py_cache).rglob("*.py")):
            try:
                content = py_file.read_text(errors="ignore")
                if any(kw in content for kw in keywords):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(py_file, dest)
                    self.logger.info(
                        "Found pipeline code in %s → copied to %s", py_file, dest
                    )
                    return dest
            except Exception:
                continue

        self.logger.warning(
            "No Zero123Plus pipeline class found in any .py file from "
            "sudo-ai/zero123plus."
        )
        return None

    def _fetch_pipeline_from_github(self, dest: Path) -> Optional[Path]:
        """
        Shallow-clone SUDO-AI-3D/zero123plus from GitHub, search every .py file
        for the Zero123PlusPipeline class, copy the matching file to *dest*, and
        return *dest* on success.  Returns None on any failure.
        """
        import shutil
        import subprocess
        import tempfile

        github_url = "https://github.com/SUDO-AI-3D/zero123plus.git"
        self.logger.info(
            "Attempting shallow clone of %s to find pipeline code …", github_url
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["git", "clone", "--depth=1", github_url, tmp],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                self.logger.warning(
                    "git clone of %s failed (rc=%d): %s",
                    github_url, result.returncode, result.stderr.strip(),
                )
                return None

            keywords = ("Zero123PlusPipeline", "class Zero123Plus", "zero123plus")
            for py_file in sorted(Path(tmp).rglob("*.py")):
                try:
                    content = py_file.read_text(errors="ignore")
                    if any(kw in content for kw in keywords):
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(py_file, dest)
                        self.logger.info(
                            "Found pipeline code in %s → copied to %s", py_file, dest
                        )
                        return dest
                except Exception:
                    continue

        self.logger.warning(
            "No Zero123Plus pipeline class found in SUDO-AI-3D/zero123plus clone."
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
