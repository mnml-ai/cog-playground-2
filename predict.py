import os
import re
import time
from typing import List, Dict, Optional
import requests
from urllib.parse import urlparse

import torch
import numpy as np
import random
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
    DPMSolverSDEScheduler
)
from compel import Compel
from cog import BasePredictor, Input, Path
import logging

# Constants
MODEL_CACHE = "model_cache"
MAX_CACHED_MODELS = 3
DEVICE = "cuda"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)

class Predictor(BasePredictor):
    def setup(self):
        """Initialize the predictor."""
        self.model_pipes: Dict[str, Dict] = {}
        os.makedirs(MODEL_CACHE, exist_ok=True)

    def load_model(self, model_name: str, model_url: str = None):
        """Load a specific model if it's not already loaded."""
        model_key = model_url or model_name
        if model_key in self.model_pipes:
            logger.info(f"Using cached model: {model_key}")
            self.model_pipes[model_key]['last_used'] = time.time()
            return self.model_pipes[model_key]['pipe']

        self._manage_model_cache()

        if model_url:
            pipe = self._load_custom_model(model_url)
        else:
            pipe = self._load_predefined_model(model_name)

        self._setup_pipeline(pipe)
        self._init_compel(pipe)
        self.model_pipes[model_key] = {'pipe': pipe, 'last_used': time.time()}
        return pipe

    def _init_compel(self, pipe):
        self.compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

    def _manage_model_cache(self):
        """Manage the model cache to ensure it doesn't exceed the maximum size."""
        if len(self.model_pipes) >= MAX_CACHED_MODELS:
            lru_model = min(self.model_pipes, key=lambda k: self.model_pipes[k]['last_used'])
            logger.info(f"Removing least recently used model from cache: {lru_model}")
            del self.model_pipes[lru_model]

    def _load_custom_model(self, model_url: str):
        """Load a custom model from a URL."""
        model_path = self._download_model(model_url)
        logger.info(f"Loading custom model from {model_path}")
        try:
            return StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            ).to(DEVICE)
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            raise ValueError(f"Failed to load custom model from {model_url}")

    def _load_predefined_model(self, model_name: str):
        """Load a predefined model."""
        logger.info(f"Loading {model_name} pipeline...")
        model_id_map = {
            "stable-diffusion-v1-5": ("runwayml/stable-diffusion-v1-5", None),
            "realistic-vision-v5-1": ("SG161222/Realistic_Vision_V5.1_noVAE", None),
            "realistic-vision-v6-0b": ("SG161222/Realistic_Vision_V6.0_B1_noVAE", {
                "safety_checker": None,
                "requires_safety_checker": False
            }),
        }
        
        if model_name not in model_id_map:
            raise ValueError(f"Unknown model name: {model_name}")
        
        model_id, custom_config = model_id_map[model_name]
        
        try:
            return StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                cache_dir=MODEL_CACHE,
                revision="main",
                **(custom_config or {})
            ).to(DEVICE)
        except Exception as e:
            logger.warning(f"Failed to load with safetensors: {e}")
            logger.info("Attempting to load with PyTorch weights...")
            return StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=False,
                cache_dir=MODEL_CACHE,
                revision="main",
                **(custom_config or {})
            ).to(DEVICE)

    def _setup_pipeline(self, pipe):
        """Set up the pipeline with optimizations."""
        if pipe.vae is None:
            logger.info("VAE not found in the pipeline. Loading a default VAE.")
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=torch.float16,
                cache_dir=MODEL_CACHE
            ).to(DEVICE)
            pipe.vae = vae
        
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers enabled successfully")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")
            logger.info("Falling back to default attention mechanism")
        
        pipe.enable_model_cpu_offload()

    def _download_model(self, url: str) -> str:
        """Download a model file from a URL."""
        if 'civitai.com' in url:
            url = self._extract_civitai_download_url(url)

        local_filename = os.path.join(MODEL_CACHE, os.path.basename(urlparse(url).path))
        
        if os.path.exists(local_filename):
            logger.info(f"Model file already exists: {local_filename}")
            return local_filename

        logger.info(f"Downloading model from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return local_filename

    def _extract_civitai_download_url(self, url: str) -> str:
        """Extract the download URL from a civitai.com page."""
        logger.info(f"Detected civitai.com URL: {url}")
        response = requests.get(url)
        response.raise_for_status()
        match = re.search(r'href="(https://civitai.com/api/download/models/\d+)"', response.text)
        if match:
            return match.group(1)
        raise ValueError("Couldn't find the download link in the civitai.com page.")

    def _set_seed(self, seed: int):
        """Set seed for all random number generators."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _get_views(self, height: int, width: int, window_size: int = 64, stride: int = 8):
        """Generate views for MultiDiffusion."""
        views = []
        for h in range(0, height - window_size + 1, stride):
            for w in range(0, width - window_size + 1, stride):
                views.append((h, w, h + window_size, w + window_size))
        return views

    def _multi_diffusion(self, pipe, latents, prompt_embeds, negative_prompt_embeds, num_inference_steps, guidance_scale, views):
        """Apply MultiDiffusion technique."""
        for step in pipe.progress_bar(pipe.scheduler.timesteps):
            latent_model_input = pipe.scheduler.scale_model_input(latents, step)
            noise_pred = torch.zeros_like(latents)
            weights_sum = torch.zeros_like(latents)

            for view in views:
                h_start, w_start, h_end, w_end = view
                latent_view = latent_model_input[:, :, h_start:h_end, w_start:w_end]
                
                noise_pred_view = pipe.unet(
                    latent_view,
                    step,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )[0]

                if guidance_scale > 1.0:
                    uncond_noise_pred_view = pipe.unet(
                        latent_view,
                        step,
                        encoder_hidden_states=negative_prompt_embeds,
                        cross_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    noise_pred_view = uncond_noise_pred_view + guidance_scale * (noise_pred_view - uncond_noise_pred_view)

                noise_pred[:, :, h_start:h_end, w_start:w_end] += noise_pred_view
                weights_sum[:, :, h_start:h_end, w_start:w_end] += 1

            noise_pred = noise_pred / weights_sum
            latents = pipe.scheduler.step(noise_pred, step, latents, return_dict=False)[0]

        return latents

    @torch.inference_mode()
    def predict(
        self,
        model: str = Input(description="Choose a model", choices=["stable-diffusion-v1-5", "realistic-vision-v5-1", "realistic-vision-v6-0b"], default="stable-diffusion-v1-5"),
        model_url: str = Input(description="URL to a custom model file (optional)", default=None),
        prompt: str = Input(description="Input prompt - using compel, use +++ to increase words weight"),
        negative_prompt: str = Input(
            description="Negative prompt - using compel, use +++ to increase words weight.",
            default="lowres, cropped, worst quality, low quality"
        ),
        width: int = Input(description="Width of output image", ge=64, le=2048, default=512),
        height: int = Input(description="Height of output image", ge=64, le=2048, default=512),
        num_outputs: int = Input(description="Number of images to output", ge=1, le=4, default=1),
        num_inference_steps: int = Input(description="Number of denoising steps", ge=1, le=100, default=50),
        guidance_scale: float = Input(description="Scale for classifier-free guidance", ge=1, le=20, default=7.5),
        scheduler: str = Input(description="Choose a scheduler", choices=["DDIM", "Euler a", "DPM++ 2M Karras", "DPM++ 3M SDE Karras"], default="DPM++ 3M SDE Karras"),
        seed: int = Input(description="Random seed. Leave blank to randomize the seed", default=None),
        use_multi_diffusion: bool = Input(description="Use MultiDiffusion for enhancement", default=True),
        window_size: int = Input(description="Window size for MultiDiffusion", ge=32, le=128, default=64),
        stride: int = Input(description="Stride for MultiDiffusion", ge=4, le=64, default=8),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        pipe = self.load_model(model, model_url)

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        logger.info(f"Using seed: {seed}")
        self._set_seed(seed)

        if width * height > 1024 * 1024:
            logger.warning(f"Large image size ({width}x{height}). This may cause out-of-memory errors on some systems.")

        pipe.scheduler = self._get_scheduler(scheduler, pipe.scheduler.config)
        generator = torch.Generator(DEVICE).manual_seed(seed)

        # Process prompts with Compel
        prompt_embeds = self.compel_proc(prompt)
        negative_prompt_embeds = self.compel_proc(negative_prompt) if negative_prompt else None

        # Generate initial latents
        latents = torch.randn(
            (1, pipe.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=DEVICE,
            dtype=torch.float16
        )

        # Apply MultiDiffusion if enabled
        if use_multi_diffusion:
            views = self._get_views(height // 8, width // 8, window_size // 8, stride // 8)
            latents = self._multi_diffusion(pipe, latents, prompt_embeds, negative_prompt_embeds, num_inference_steps, guidance_scale, views)
        else:
            # Use standard diffusion process
            latents = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_outputs,
                generator=generator,
                latents=latents,
                output_type="latent"
            ).images

        # Decode latents to images
        images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return self._save_output_images(pil_images)
    
    def _get_scheduler(self, scheduler_name: str, config):
        """Get the specified scheduler."""
        schedulers = {
            "DDIM": DDIMScheduler,
            "Euler a": EulerAncestralDiscreteScheduler,
            "DPM++ 2M Karras": lambda config: DPMSolverMultistepScheduler.from_config(config, algorithm_type="dpmsolver++", use_karras_sigmas=True),
            "DPM++ 3M SDE Karras": lambda config: DPMSolverSDEScheduler.from_config(config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
        }
        
        scheduler_class = schedulers.get(scheduler_name)
        if not scheduler_class:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        try:
            return scheduler_class.from_config(config) if scheduler_class in [DDIMScheduler, EulerAncestralDiscreteScheduler] else scheduler_class(config)
        except Exception as e:
            logger.error(f"Failed to initialize {scheduler_name} scheduler: {e}")
            logger.info("Falling back to DPM++ 2M Karras.")
            return DPMSolverMultistepScheduler.from_config(config, algorithm_type="dpmsolver++", use_karras_sigmas=True)

    def _save_output_images(self, images) -> List[Path]:
        """Save the output images and return their paths."""
        output_paths = []
        for i, image in enumerate(images):
            output_path = f"/tmp/output_{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))
        return output_paths