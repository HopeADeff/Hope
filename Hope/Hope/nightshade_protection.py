
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

try:
    from diffusers import StableDiffusionPipeline
    from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    from utils import println, ensure_utf8_stdout, validate_image_path, get_model_path
    from gpu_utils import get_device
except ImportError:
    pass

class NightshadeProtector:
    def __init__(
        self,
        source_concept: str = "artwork",
        target_concept: str = "noise",
        intensity: float = 0.08,
        iterations: int = 50,
        verbose: bool = True
    ):
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("Diffusers library not found. Please install requirements.")
            
        self.device = get_device()
        self.source_concept = source_concept
        self.target_concept = target_concept
        self.intensity = intensity
        self.iterations = iterations
        self.verbose = verbose
        
        if verbose:
            println("=== NIGHTSHADE (SD v1.5) PROTECTOR ===")
            println(f"Device: {self.device}")
            println(f"Poisoning: '{source_concept}' -> '{target_concept}'")

        model_path = get_model_path("sd-v1-5")
        if verbose:
            println(f"Loading models from: {model_path}")
            
        try:
            self.vae = AutoencoderKL.from_pretrained(
                model_path, subfolder="vae", local_files_only=True
            ).to(self.device)
            
            self.tokenizer = CLIPTokenizer.from_pretrained(
                model_path, subfolder="tokenizer", local_files_only=True
            )
            
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_path, subfolder="text_encoder", local_files_only=True
            ).to(self.device)
            
            self.unet = UNet2DConditionModel.from_pretrained(
                model_path, subfolder="unet", local_files_only=True
            ).to(self.device)
            
            self.scheduler = PNDMScheduler.from_pretrained(
                model_path, subfolder="scheduler", local_files_only=True
            )
            
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            self.unet.requires_grad_(False)
            
            if verbose:
                println("Models loaded successfully.")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load models. Did you run download_models.py?\n{e}")

    def _get_text_embedding(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(
            prompt, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        with torch.no_grad():
            embeddings = self.text_encoder(tokens)[0]
        return embeddings

    def fit_image_to_concept(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            orig_latents = self.vae.encode(image_tensor.to(self.device)).latent_dist.sample()
            orig_latents = orig_latents * 0.18215
        
        perturbed_latents = orig_latents.clone().detach()
        perturbed_latents.requires_grad = True
        
        optimizer = torch.optim.Adam([perturbed_latents], lr=0.01)
        
        source_emb = self._get_text_embedding(f"a photo of {self.source_concept}")
        target_emb = self._get_text_embedding(f"a photo of {self.target_concept}")
        uncond_emb = self._get_text_embedding("")
        
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, 
            (1,), device=self.device
        )
        
        pbar = tqdm(range(self.iterations), disable=not self.verbose)
        for i in pbar:
            noise = torch.randn_like(perturbed_latents)
            noisy_latents = self.scheduler.add_noise(perturbed_latents, noise, timesteps)
            
            noise_pred_source = self.unet(noisy_latents, timesteps, encoder_hidden_states=source_emb).sample
            
            with torch.no_grad():
                noise_pred_target = self.unet(noisy_latents, timesteps, encoder_hidden_states=target_emb).sample
            
            loss = F.mse_loss(noise_pred_source, noise_pred_target)
            
            reg_loss = F.mse_loss(perturbed_latents, orig_latents) * 10.0
            
            total_loss = loss + reg_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if self.verbose:
                if i % 1 == 0:
                    println(f"STATUS: Iter {i+1}/{self.iterations}")

        with torch.no_grad():
            perturbed_latents = 1 / 0.18215 * perturbed_latents
            image = self.vae.decode(perturbed_latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            
        return image

    def protect_image(
        self,
        input_path: str,
        output_path: str,
        output_quality: int = 92
    ) -> bool:
        try:
            validate_image_path(input_path)
            
            img = Image.open(input_path).convert('RGB').resize((512, 512)) 
            img_tensor = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0 
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            
            if self.verbose:
                println("Running Nightshade optimization...")
            
            protected_tensor = self.fit_image_to_concept(img_tensor)
            
            protected_array = protected_tensor.cpu().permute(0, 2, 3, 1).numpy()[0]
            protected_img = Image.fromarray((protected_array * 255).astype(np.uint8))
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            protected_img.save(output_path, quality=output_quality)
            
            if self.verbose:
                println("Nightshade protection complete.")
                
            return True
            
        except Exception as e:
            println(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False

def protect_image(**kwargs):
    protector = NightshadeProtector(**kwargs)
    return protector.protect_image(kwargs['input_path'], kwargs['output_path'])
