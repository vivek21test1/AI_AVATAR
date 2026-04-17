"""
Zero123++ Pipeline — bundled vendored implementation.

Zero123PlusPipeline: single-image to 6-view multi-view generation.

Model : sudo-ai/zero123plus-v1.1
Paper : https://arxiv.org/abs/2310.15110  (SUDO-AI-3D, 2023)
License: Apache 2.0

This file is bundled with the project because the original HuggingFace repo
(sudo-ai/zero123plus) has been deleted and the GitHub raw URLs are no longer
accessible from the deployment environment.

Component names in sudo-ai/zero123plus-v1.1/model_index.json
-------------------------------------------------------------
Required : vae, unet, scheduler
Optional : vision_encoder, feature_extractor_clip, feature_extractor_vae,
           cc_projection, text_encoder, tokenizer, safety_checker
Config   : ramping_coefficients  (list of floats — NOT a loadable module)
"""

from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers import (
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)


class Zero123PlusPipeline(DiffusionPipeline):
    """
    Pipeline for zero-shot single-image to 6-view multi-view generation.

    Generates 6 consistent multi-view images arranged in a 3 × 2 grid.
    Input should be a clean-background RGB/RGBA image.

    Usage::

        pipe = Zero123PlusPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.1",
            custom_pipeline="path/to/pipeline_zero123plus.py",
            torch_dtype=torch.float16,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        output_grid = pipe(input_image, num_inference_steps=75).images[0]
        # output_grid is PIL Image of size (width*3, height*2)
    """

    # All non-core components are optional — diffusers uses default=None in
    # __init__ to decide what's required vs optional.
    _optional_components = [
        "vision_encoder",
        "feature_extractor_clip",
        "feature_extractor_vae",
        "cc_projection",
        "image_projection_model",  # legacy name variant
        "image_encoder",           # legacy name variant
        "feature_extractor",       # legacy name variant
        "text_encoder",
        "tokenizer",
        "safety_checker",
    ]

    def __init__(
        self,
        # ── Required components (always present in model_index.json) ──────
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: EulerAncestralDiscreteScheduler,
        # ── Optional components — zero123plus-v1.1 actual names ──────────
        vision_encoder: Optional[CLIPVisionModelWithProjection] = None,
        feature_extractor_clip: Optional[CLIPImageProcessor] = None,
        feature_extractor_vae: Optional[CLIPImageProcessor] = None,
        # ── Projection layer (may or may not exist) ───────────────────────
        cc_projection=None,
        # ── Legacy naming aliases (kept for older model_index.json) ───────
        image_encoder: Optional[CLIPVisionModelWithProjection] = None,
        feature_extractor: Optional[CLIPImageProcessor] = None,
        image_projection_model=None,
        # ── Components present in model_index but not used at inference ───
        text_encoder=None,
        tokenizer=None,
        safety_checker=None,
        # NOTE: no **kwargs — diffusers introspects __init__ and would treat
        # 'kwargs' as a required component directory.  Unrecognised config
        # values (e.g. ramping_coefficients) are stored automatically in
        # self.config by DiffusionPipeline.from_pretrained.
    ):
        super().__init__()

        # Every __init__ parameter MUST be passed to register_modules — even
        # as None.  diffusers' .components property validates that all
        # parameters declared in __init__ have been set as attributes via
        # register_modules, and raises ValueError for any that are missing.
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            # v1.1 actual names — carry the real object when provided
            vision_encoder=vision_encoder,
            feature_extractor_clip=feature_extractor_clip,
            feature_extractor_vae=feature_extractor_vae,
            # legacy aliases — None when the v1.1 name was used instead
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            # projection (only one will be non-None at a time)
            cc_projection=cc_projection,
            image_projection_model=image_projection_model,
            # unused but declared so .components validation passes
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            safety_checker=safety_checker,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_vision_encoder(self):
        """Return whichever vision encoder was registered (either name)."""
        enc = getattr(self, "vision_encoder", None) or getattr(self, "image_encoder", None)
        if enc is None:
            raise RuntimeError(
                "Zero123PlusPipeline has no vision encoder. "
                "Expected 'vision_encoder' or 'image_encoder' component."
            )
        return enc

    def _get_feature_extractor_clip(self):
        """Return whichever CLIP feature extractor was registered (either name)."""
        fe = (
            getattr(self, "feature_extractor_clip", None)
            or getattr(self, "feature_extractor", None)
        )
        if fe is None:
            raise RuntimeError(
                "Zero123PlusPipeline has no CLIP feature extractor. "
                "Expected 'feature_extractor_clip' or 'feature_extractor' component."
            )
        return fe

    def _get_projection(self):
        """Return projection layer if any (either name), or None."""
        return (
            getattr(self, "cc_projection", None)
            or getattr(self, "image_projection_model", None)
        )

    def _encode_image_clip(
        self, image: Image.Image, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Encode reference image through CLIP → cross-attention hidden states."""
        feature_extractor = self._get_feature_extractor_clip()
        vision_encoder = self._get_vision_encoder()

        pixel_values = feature_extractor(
            images=image, return_tensors="pt"
        ).pixel_values.to(device=device, dtype=dtype)

        image_embeds = vision_encoder(pixel_values).image_embeds  # [1, D]

        proj = self._get_projection()
        if proj is not None:
            image_embeds = proj(image_embeds)          # [1, N, cross_dim]
        else:
            image_embeds = image_embeds.unsqueeze(1)   # [1, 1, D]

        return image_embeds

    def _encode_image_vae(
        self,
        image: Image.Image,
        width: int,
        height: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode reference image through VAE (used when UNet expects 8 in-channels)."""
        img = image.convert("RGB").resize((width, height), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        t = (2.0 * t - 1.0).to(device=device, dtype=dtype)
        latent = self.vae.encode(t).latent_dist.mode()
        return latent * self.vae.config.scaling_factor

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        image: Union[Image.Image, torch.FloatTensor],
        width: int = 320,
        height: int = 320,
        num_inference_steps: int = 75,
        guidance_scale: float = 4.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        **kwargs,
    ) -> ImagePipelineOutput:
        """
        Args:
            image: Input reference PIL Image.
            width / height: Per-view resolution in pixels (default 320 × 320).
            num_inference_steps: Diffusion denoising steps (default 75).
            guidance_scale: Classifier-free guidance scale (default 4.0).
            generator: Optional ``torch.Generator`` for reproducibility.
            latents: Optional pre-sampled latents; random if None.
            output_type: ``"pil"`` or ``"np"``.

        Returns:
            :class:`~diffusers.ImagePipelineOutput` where ``.images[0]`` is a
            PIL Image of size ``(width*3, height*2)`` containing the 6-view
            multi-view grid.
        """
        device = self.device
        dtype = next(self.unet.parameters()).dtype

        # Accept torch tensor input — convert to PIL
        if isinstance(image, torch.Tensor):
            arr = image.cpu().float()
            if arr.ndim == 4:
                arr = arr[0]
            arr = arr.permute(1, 2, 0).numpy()
            arr = (arr * 0.5 + 0.5).clip(0, 1)
            image = Image.fromarray((arr * 255).astype(np.uint8))

        # 1. Encode reference image (CLIP)
        image_embeds = self._encode_image_clip(image, device, dtype)  # [1, N, D]

        do_cfg = guidance_scale > 1.0
        if do_cfg:
            neg = torch.zeros_like(image_embeds)
            image_embeds_in = torch.cat([neg, image_embeds])  # [2, N, D]
        else:
            image_embeds_in = image_embeds

        # 2. Prepare latents for the full 3×2 output grid
        out_w = width * 3
        out_h = height * 2
        lat_w = out_w // self.vae_scale_factor
        lat_h = out_h // self.vae_scale_factor
        in_ch = self.unet.config.in_channels

        # in_channels == 8: UNet expects concat(noisy_latent, ref_latent)
        # in_channels == 4: pure cross-attention conditioning
        noise_ch = 4 if in_ch == 8 else in_ch

        if latents is None:
            latents = randn_tensor(
                (1, noise_ch, lat_h, lat_w),
                generator=generator,
                device=device,
                dtype=dtype,
            )
        else:
            latents = latents.to(device=device, dtype=dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = latents * self.scheduler.init_noise_sigma

        # Spatial reference latent (only used when in_ch == 8)
        ref_latent = None
        if in_ch == 8:
            ref_latent = self._encode_image_vae(image, width, height, device, dtype)
            ref_latent = ref_latent.repeat(1, 1, 2, 3)  # tile to 3×2 grid size

        # 3. Denoising loop
        for t in self.progress_bar(self.scheduler.timesteps):
            lat_in = torch.cat([latents] * 2) if do_cfg else latents
            lat_in = self.scheduler.scale_model_input(lat_in, t)

            if ref_latent is not None:
                ref_in = torch.cat([ref_latent] * 2) if do_cfg else ref_latent
                lat_in = torch.cat([lat_in, ref_in], dim=1)

            noise_pred = self.unet(
                lat_in,
                t,
                encoder_hidden_states=image_embeds_in,
            ).sample

            if do_cfg:
                pred_uncond, pred_cond = noise_pred.chunk(2)
                noise_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            latents = self.scheduler.step(
                noise_pred, t, latents, generator=generator
            ).prev_sample

        # 4. Decode latents → pixel space
        img_tensor = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        img_tensor = (img_tensor / 2 + 0.5).clamp(0, 1)
        img_np = img_tensor.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            images: List = [
                Image.fromarray((img * 255).astype(np.uint8)) for img in img_np
            ]
        else:
            images = list(img_np)

        return ImagePipelineOutput(images=images)
