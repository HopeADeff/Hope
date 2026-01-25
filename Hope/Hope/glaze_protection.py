
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

try:
    from diffusers import (
        AutoencoderKL,
        UNet2DConditionModel,
        PNDMScheduler,
        EulerDiscreteScheduler
    )
    from transformers import (
        CLIPTextModel,
        CLIPTokenizer,
        CLIPTextModelWithProjection
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    from utils import (
        println,
        ensure_utf8_stdout,
        validate_image_path,
        get_model_path,
        check_image_dimensions
    )
    from gpu_utils import get_device
except ImportError:
    pass

class GlazeStyleProtector:
    STYLE_PROMPTS = {
        "abstract": "abstract expressionist painting, chaotic, splatters, jackson pollock",
        "impressionist": "impressionist painting, claude monet, visible brushstrokes, light and color",
        "cubist": "cubist painting, picasso, geometric shapes, fragmented perspective",
        "sketch": "charcoal sketch, rough pencil drawing, monochrome, hatching",
        "watercolor": "watercolor painting, wet on wet, flowing pigments, soft edges"
    }

    def __init__(
        self,
        target_style: str = "abstract",
        intensity: float = 0.45,
        iterations: int = 50,
        verbose: bool = True,
        model_type: str = "sd-v1-5"
    ):
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("Diffusers library not found.")

        if target_style not in self.STYLE_PROMPTS:
            raise ValueError(f"Unknown style: {target_style}")

        if model_type not in ["sd-v1-5", "sdxl"]:
            raise ValueError(
                f"Unsupported model_type: {model_type}. Use 'sd-v1-5' or 'sdxl'"
            )

        self.device = get_device()
        self.target_style = target_style
        self.style_prompt = self.STYLE_PROMPTS[target_style]
        self.intensity = intensity
        self.iterations = iterations
        self.verbose = verbose
        self.model_type = model_type

        if model_type == "sdxl":
            self.vae_scale = 0.13025
            self.target_resolution = 1024
            model_name_display = "SDXL"
            default_hf_model = "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            self.vae_scale = 0.18215
            self.target_resolution = 512
            model_name_display = "SD v1.5"
            default_hf_model = "runwayml/stable-diffusion-v1-5"
        
        if verbose:
            println(f"=== GLAZE ({model_name_display}) PROTECTOR ===")
            println(f"Device: {self.device}")
            println(f"Target Style: '{target_style}'")

        model_path = get_model_path(model_type)
        local_files_only = True

        if not os.path.exists(model_path) or not (
            Path(model_path) / "model_index.json"
        ).exists():
            if verbose:
                println(
                    f"Local model not found at {model_path}. "
                    f"Fallback to HuggingFace ({default_hf_model})"
                )
            model_path = default_hf_model
            local_files_only = False

        if verbose:
            println(
                f"Loading models from: {model_path} (local_only={local_files_only})"
            )

        try:
            self.vae = AutoencoderKL.from_pretrained(
                model_path, subfolder="vae", local_files_only=local_files_only
            ).to(self.device)
            
            self.tokenizer = CLIPTokenizer.from_pretrained(
                model_path,
                subfolder="tokenizer",
                local_files_only=local_files_only
            )

            if model_type == "sdxl":
                self.text_encoder = CLIPTextModel.from_pretrained(
                    model_path,
                    subfolder="text_encoder",
                    local_files_only=local_files_only
                ).to(self.device)
                self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                    model_path,
                    subfolder="text_encoder_2",
                    local_files_only=local_files_only
                ).to(self.device)
                self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                    model_path,
                    subfolder="tokenizer_2",
                    local_files_only=local_files_only
                )
            else:
                self.text_encoder = CLIPTextModel.from_pretrained(
                    model_path,
                    subfolder="text_encoder",
                    local_files_only=local_files_only
                ).to(self.device)
                self.text_encoder_2 = None
                self.tokenizer_2 = None

            self.unet = UNet2DConditionModel.from_pretrained(
                model_path,
                subfolder="unet",
                local_files_only=local_files_only
            ).to(self.device)

            if model_type == "sdxl":
                self.scheduler = EulerDiscreteScheduler.from_pretrained(
                    model_path,
                    subfolder="scheduler",
                    local_files_only=local_files_only
                )
            else:
                self.scheduler = PNDMScheduler.from_pretrained(
                    model_path,
                    subfolder="scheduler",
                    local_files_only=local_files_only
                )

            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            if self.text_encoder_2 is not None:
                self.text_encoder_2.requires_grad_(False)
            self.unet.requires_grad_(False)

        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")

    def _get_text_embedding(self, prompt: str):
        if self.model_type == "sdxl":
            tokens = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)
            tokens_2 = self.tokenizer_2(
                prompt,
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)

            with torch.no_grad():
                encoder_output_1 = self.text_encoder(
                    tokens, output_hidden_states=True
                )
                encoder_output_2 = self.text_encoder_2(
                    tokens_2, output_hidden_states=True
                )

                prompt_embeds = encoder_output_1.hidden_states[-2]
                pooled_projections = encoder_output_2.text_embeds

                return (prompt_embeds, pooled_projections)
        else:
            tokens = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)
            with torch.no_grad():
                return self.text_encoder(tokens)[0]

    def protect_image(
        self,
        input_path: str,
        output_path: str,
        output_quality: int = 92
    ) -> bool:
        try:
            validate_image_path(input_path)

            img = Image.open(input_path).convert('RGB')
            original_size = img.size

            ratio = min(
                self.target_resolution / original_size[0],
                self.target_resolution / original_size[1]
            )
            new_width = int(original_size[0] * ratio)
            new_height = int(original_size[1] * ratio)
            
            new_width = (new_width // 8) * 8
            new_height = (new_height // 8) * 8

            img = img.resize((new_width, new_height), Image.LANCZOS)

            img_tensor = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                latents = self.vae.encode(img_tensor).latent_dist.sample()
                latents = latents * self.vae_scale

            optimized_latents = latents.clone().detach()
            optimized_latents.requires_grad = True
            optimizer = torch.optim.Adam([optimized_latents], lr=0.02)

            style_emb_result = self._get_text_embedding(self.style_prompt)

            timesteps = torch.tensor([100], device=self.device)

            if self.verbose:
                println("Optimizing style cloak...")

            for i in tqdm(range(self.iterations), disable=not self.verbose):
                noise = torch.randn_like(optimized_latents)
                noisy_latents = self.scheduler.add_noise(optimized_latents, noise, timesteps)
                
                if self.model_type == "sdxl":
                    prompt_embeds, pooled_projections = style_emb_result

                    batch_size = noisy_latents.shape[0]
                    height = noisy_latents.shape[2] * 8
                    width = noisy_latents.shape[3] * 8
                    time_ids = torch.tensor(
                        [[height, width, 0, 0, 0, 0]] * batch_size,
                        dtype=torch.float32,
                        device=self.device
                    )

                    added_cond_kwargs = {
                        "text_embeds": pooled_projections,
                        "time_ids": time_ids
                    }
                    noise_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs
                    ).sample
                else:
                    noise_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=style_emb_result
                    ).sample

                loss_style = F.mse_loss(noise_pred, noise)

                loss_content = F.mse_loss(optimized_latents, latents) * 5.0

                total_loss = loss_style * self.intensity + loss_content
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if self.verbose:
                    println(f"STATUS: Iter {i+1}/{self.iterations}")
            
            with torch.no_grad():
                rec_latents = 1 / self.vae_scale * latents
                rec_decoded = self.vae.decode(rec_latents).sample
                rec_decoded = (rec_decoded / 2 + 0.5).clamp(0, 1)

                protected_latents = 1 / self.vae_scale * optimized_latents
                prot_decoded = self.vae.decode(protected_latents).sample
                prot_decoded = (prot_decoded / 2 + 0.5).clamp(0, 1)

            rec_array = rec_decoded.cpu().permute(0, 2, 3, 1).numpy()[0]
            prot_array = prot_decoded.cpu().permute(0, 2, 3, 1).numpy()[0]

            delta = prot_array - rec_array
            delta_img = Image.fromarray(
                (delta * 127.5 + 127.5).astype(np.uint8)
            )
            delta_img = delta_img.resize(original_size, Image.LANCZOS)
            delta_upscaled = (np.array(delta_img).astype(np.float32) / 127.5) - 1.0
            original_img = Image.open(input_path).convert('RGB')
            original_array = np.array(original_img).astype(np.float32) / 255.0
            final_array = original_array + delta_upscaled
            final_array = np.clip(final_array, 0, 1)
            protected_img = Image.fromarray((final_array * 255).astype(np.uint8))

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            protected_img.save(output_path, quality=output_quality)

            if self.verbose:
                println("Glaze protection complete.")
            return True

        except Exception as e:
            println(f"Error: {e}")
            return False

def protect_image(**kwargs):
    protector = GlazeStyleProtector(**kwargs)
    return protector.protect_image(kwargs['input_path'], kwargs['output_path'])
