"""
Static registry of all supported 3-D avatar models.

Each entry maps a short key (used in API calls) to its configuration:
  hf_repo     – HuggingFace model-id for snapshot_download
  local_dir   – where weights are cached after download
  github_repo – source-code repo to clone (inference code)
  repo_dir    – local clone path
  key_files   – files inside local_dir that prove a complete download;
                when empty, any *.safetensors / *.bin / *.ckpt / *.pth counts
  vram_gb     – minimum GPU VRAM required
  output_format – primary output ("obj", "png", "ply")
"""

from app.core.config import settings

MODEL_REGISTRY: dict[str, dict] = {
    "triposr": {
        "display_name": "TripoSR",
        "hf_repo": "stabilityai/TripoSR",
        "local_dir": str(settings.MODEL_CACHE_DIR / "triposr"),
        "github_repo": "https://github.com/VAST-AI-Research/TripoSR.git",
        "repo_dir": str(settings.REPOS_DIR / "TripoSR"),
        "key_files": ["config.yaml", "model.ckpt"],
        "vram_gb": 6,
        "output_format": "obj",
        "description": "Single image → Textured 3D mesh (.obj) in ~0.5 s",
        "license": "MIT",
        "commercial_use": True,
    },
    "zero123plus": {
        "display_name": "Zero123++",
        "hf_repo": "sudo-ai/zero123plus",
        "local_dir": str(settings.MODEL_CACHE_DIR / "zero123plus"),
        "github_repo": None,
        "repo_dir": None,
        "key_files": ["pipeline_zero123plus.py"],
        "vram_gb": 5,
        "output_format": "png",
        "description": "Single image → 6 consistent multi-view images",
        "license": "Apache-2.0",
        "commercial_use": True,
    },
    "crm": {
        "display_name": "CRM",
        "hf_repo": "Zhengyi/CRM",
        "local_dir": str(settings.MODEL_CACHE_DIR / "crm"),
        "github_repo": "https://github.com/thu-ml/CRM.git",
        "repo_dir": str(settings.REPOS_DIR / "CRM"),
        "key_files": [],
        "vram_gb": 8,
        "output_format": "obj",
        "description": "Single image → Full textured 3D mesh in ~10 s",
        "license": "MIT",
        "commercial_use": True,
    },
    "wonder3d": {
        "display_name": "Wonder3D",
        "hf_repo": "flamehaze1115/wonder3d-v1.0",
        "local_dir": str(settings.MODEL_CACHE_DIR / "wonder3d"),
        "github_repo": "https://github.com/xxlong0/Wonder3D.git",
        "repo_dir": str(settings.REPOS_DIR / "Wonder3D"),
        "key_files": [],
        "vram_gb": 8,
        "output_format": "obj",
        "description": "Single image → Multi-view normals + detailed 3D mesh in 2–3 min",
        "license": "MIT",
        "commercial_use": True,
    },
    "instantmesh": {
        "display_name": "InstantMesh",
        "hf_repo": "TencentARC/InstantMesh",
        "local_dir": str(settings.MODEL_CACHE_DIR / "instantmesh"),
        "github_repo": "https://github.com/TencentARC/InstantMesh.git",
        "repo_dir": str(settings.REPOS_DIR / "InstantMesh"),
        "key_files": [],
        "vram_gb": 16,
        "output_format": "obj",
        "description": "Single image → High-quality textured 3D mesh; ComfyUI support",
        "license": "Apache-2.0",
        "commercial_use": True,
    },
    "lam": {
        "display_name": "LAM (Large Avatar Model)",
        "hf_repo": "aigc3d/LAM",
        "local_dir": str(settings.MODEL_CACHE_DIR / "lam"),
        "github_repo": "https://github.com/aigc3d/LAM.git",
        "repo_dir": str(settings.REPOS_DIR / "LAM"),
        "key_files": [],
        "vram_gb": 16,
        "output_format": "ply",
        "description": "Single portrait → Animatable 3D Gaussian head avatar (90 FPS real-time)",
        "license": "Apache-2.0",
        "commercial_use": True,
    },
}
