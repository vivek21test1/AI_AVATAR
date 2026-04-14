"""
CRM (Convolutional Reconstruction Model) wrapper.

HF repo : Zhengyi/CRM (weights)  |  Code : https://github.com/thu-ml/CRM
VRAM    : 8 GB (CUDA required — nvdiffrast)
Output  : Textured mesh (mesh.obj + mesh.mtl + mesh.png) and mesh.glb

Upstream layout (clone into repos/CRM/):
  configs/nf7_v3_SNR_rd_size_stroke.yaml  — stage-1 pixel diffusion
  configs/stage2-v2-snr.yaml               — stage-2 CCM diffusion
  configs/specs_objaverse_total.json       — CRM UNet specs

Weights (HF snapshot in model_cache/crm/ or hf_hub_download):
  CRM.pth, pixel-diffusion.pth, ccm-diffusion.pth
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from app.avatar_models.base import BaseAvatarModel


def _load_repo_script(repo: Path, relative_py: str, module_name: str):
    """
    Load a single ``*.py`` from the CRM clone as an isolated module name.

    Avoids shadowing PyPI packages named ``inference``, ``model``, ``pipelines``, …
    which would otherwise win on ``sys.path`` or sit in ``sys.modules``.
    """
    path = repo / relative_py
    if not path.is_file():
        raise FileNotFoundError(f"CRM repo missing {relative_py}: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def _chdir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class CRMModel(BaseAvatarModel):

    def __init__(self, local_dir: str, repo_dir: str, device: str = "auto") -> None:
        super().__init__("crm", local_dir, device)
        self.repo_dir = Path(repo_dir)
        self._pipeline = None
        self._crm_model = None
        self._generate3d = None
        self._preprocess_image = None

    def unload(self) -> None:
        self._pipeline = None
        self._crm_model = None
        self._generate3d = None
        self._preprocess_image = None
        super().unload()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self.device != "cuda":
            try:
                import torch
                tv = torch.__version__
                cuda_ok = torch.cuda.is_available()
            except ImportError:
                tv, cuda_ok = "(not installed)", False
            raise RuntimeError(
                "CRM must run with device='cuda': mesh extraction uses nvdiffrast "
                "(RasterizeCudaContext). This process resolved the device to "
                f"{self.device!r} (torch {tv}, cuda_available={cuda_ok}). "
                "Use a GPU host with CUDA PyTorch, or pick another model."
            )

        import torch
        from huggingface_hub import hf_hub_download
        from omegaconf import OmegaConf

        repo = self.repo_dir.resolve()
        if not repo.is_dir():
            raise FileNotFoundError(
                f"CRM code repo not found at {repo}. "
                "Run model download or: git clone https://github.com/thu-ml/CRM.git "
                f"{repo}"
            )

        repo_str = str(repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        stage1_yaml = repo / "configs" / "nf7_v3_SNR_rd_size_stroke.yaml"
        stage2_yaml = repo / "configs" / "stage2-v2-snr.yaml"
        specs_path = repo / "configs" / "specs_objaverse_total.json"
        for p in (stage1_yaml, stage2_yaml, specs_path):
            if not p.is_file():
                raise FileNotFoundError(
                    f"CRM repo file missing: {p}. "
                    "Use a fresh clone of https://github.com/thu-ml/CRM (main branch)."
                )

        local = Path(self.local_dir).resolve()

        def _weight(name: str) -> str:
            direct = local / name
            if direct.is_file():
                return str(direct)
            nested = next(local.glob(f"**/{name}"), None)
            if nested is not None and nested.is_file():
                return str(nested)
            return hf_hub_download(repo_id="Zhengyi/CRM", filename=name)

        crm_pth = _weight("CRM.pth")
        pixel_pth = _weight("pixel-diffusion.pth")
        ccm_pth = _weight("ccm-diffusion.pth")

        specs = json.loads(specs_path.read_text(encoding="utf-8"))

        model_mod = _load_repo_script(repo, "model.py", "ai_avatar_thuml_crm_model")
        CRM = model_mod.CRM
        pipe_mod = _load_repo_script(repo, "pipelines.py", "ai_avatar_thuml_crm_pipelines")
        TwoStagePipeline = pipe_mod.TwoStagePipeline
        self._preprocess_image = pipe_mod.preprocess_image

        inf_mod = _load_repo_script(repo, "inference.py", "ai_avatar_thuml_crm_inference")
        self._generate3d = inf_mod.generate3d

        with _chdir(repo):
            stage1_root = OmegaConf.load(stage1_yaml).config
            stage2_root = OmegaConf.load(stage2_yaml).config

            stage1_sampler_config = stage1_root.sampler
            stage2_sampler_config = stage2_root.sampler
            stage1_model_config = stage1_root.models
            stage2_model_config = stage2_root.models
            stage1_model_config.resume = pixel_pth
            stage2_model_config.resume = ccm_pth

            self._pipeline = TwoStagePipeline(
                stage1_model_config,
                stage2_model_config,
                stage1_sampler_config,
                stage2_sampler_config,
                device=self.device,
                dtype=torch.float16,
            )

            self._crm_model = CRM(specs).to(self.device)
            self._crm_model.load_state_dict(
                torch.load(crm_pth, map_location=self.device),
                strict=False,
            )

        self._model = self._pipeline

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, image_path: str, output_dir: str, **kwargs) -> dict:
        self.ensure_loaded()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        scale: float = float(kwargs.get("scale", 5.0))
        steps: int = int(kwargs.get("steps", 50))
        seed: int = int(kwargs.get("seed", 1))

        image = Image.open(image_path).convert("RGBA")
        repo = self.repo_dir.resolve()

        with _chdir(repo):
            img_rgb = self._preprocess_image(
                image,
                "Auto Remove background",
                1.0,
                (127, 127, 127),
            )

            if hasattr(self._pipeline, "set_seed"):
                self._pipeline.set_seed(seed)

            rt_dict = self._pipeline(img_rgb, scale=scale, step=steps)
            stage1_images = rt_dict["stage1_images"]
            stage2_images = rt_dict["stage2_images"]

            np_imgs = np.concatenate(
                [np.asarray(im, dtype=np.uint8) for im in stage1_images],
                axis=1,
            )
            np_xyzs = np.concatenate(
                [np.asarray(im, dtype=np.uint8) for im in stage2_images],
                axis=1,
            )

            glb_path, obj_zip_path = self._generate3d(
                self._crm_model, np_imgs, np_xyzs, self.device
            )

        zip_p = Path(obj_zip_path)
        with zipfile.ZipFile(obj_zip_path, "r") as zf:
            member_names = [n for n in zf.namelist() if not n.endswith("/")]
            zf.extractall(out)

        obj_member = next((n for n in member_names if n.lower().endswith(".obj")), None)
        if not obj_member:
            raise RuntimeError("CRM inference ZIP did not contain a .obj mesh.")

        old_stem = Path(obj_member).stem

        # Stable names + fix MTL texture reference after rename
        if old_stem:
            for ext in (".obj", ".mtl", ".png"):
                src = out / f"{old_stem}{ext}"
                if not src.is_file():
                    continue
                dest = out / f"mesh{ext}"
                if dest.exists() and dest.resolve() != src.resolve():
                    dest.unlink()
                src.rename(dest)
            mtl = out / "mesh.mtl"
            if mtl.is_file():
                text = mtl.read_text(encoding="utf-8", errors="replace")
                text = re.sub(
                    rf"\b{re.escape(old_stem)}\.png\b",
                    "mesh.png",
                    text,
                )
                mtl.write_text(text, encoding="utf-8")

        shutil.copy2(glb_path, out / "mesh.glb")

        # Remove HuggingFace / tmp bundle next to the temp prefix (see inference.py)
        temp_parent = zip_p.parent
        temp_stem = zip_p.stem
        for ext in (".obj", ".png", ".mtl", ".zip"):
            try:
                (temp_parent / (temp_stem + ext)).unlink(missing_ok=True)
            except OSError:
                pass
        try:
            Path(glb_path).unlink(missing_ok=True)
        except OSError:
            pass

        written = sorted(
            (str(p) for p in out.iterdir() if p.is_file()),
            key=lambda s: Path(s).name,
        )
        return {
            "output_files": written,
            "output_format": "obj",
            "metadata": {"scale": scale, "steps": steps, "seed": seed},
        }
