"""
Zero123++ model wrapper.

HF repo : sudo-ai/zero123plus
VRAM    : 5 GB   |  License: Apache-2.0  |  Commercial: Yes
Output  : 6 consistent multi-view images (3×2 grid PNG + individual tiles)

The HF snapshot contains pipeline_zero123plus.py (the diffusers custom
pipeline).  We load it from disk via custom_pipeline=local_dir so the
inference runs fully offline.
"""

from pathlib import Path

from PIL import Image

from app.avatar_models.base import BaseAvatarModel


class Zero123PlusModel(BaseAvatarModel):

    def __init__(self, local_dir: str, device: str = "auto") -> None:
        super().__init__("zero123plus", local_dir, device)
        self._pipeline = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        import torch
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

        self._pipeline = DiffusionPipeline.from_pretrained(
            str(self.local_dir),
            custom_pipeline=str(self.local_dir),  # loads pipeline_zero123plus.py from disk
            torch_dtype=torch.float16,
        )
        self._pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self._pipeline.scheduler.config,
            timestep_spacing="trailing",
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

        image = Image.open(image_path).convert("RGB")
        steps: int = kwargs.get("num_inference_steps", 75)
        seed: int  = kwargs.get("seed", 42)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        grid: Image.Image = self._pipeline(
            image,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]

        grid_path = str(out / "multiview_grid.png")
        grid.save(grid_path)

        individual = self._split_grid(grid, out)

        return {
            "output_files": [grid_path] + individual,
            "output_format": "png",
            "metadata": {"num_inference_steps": steps, "views": 6},
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_grid(grid: Image.Image, out: Path) -> list[str]:
        """Split 3×2 grid (960×640) into 6 individual 320×320 tiles."""
        tile_w = grid.width  // 3
        tile_h = grid.height // 2
        paths: list[str] = []
        for row in range(2):
            for col in range(3):
                idx  = row * 3 + col
                tile = grid.crop((
                    col * tile_w, row * tile_h,
                    (col + 1) * tile_w, (row + 1) * tile_h,
                ))
                p = str(out / f"view_{idx:02d}.png")
                tile.save(p)
                paths.append(p)
        return paths
