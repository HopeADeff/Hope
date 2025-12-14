#!/usr/bin/env python3
import sys
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    from utils import ensure_utf8_stdout, println, validate_image_path, check_image_dimensions
except ImportError:
    def ensure_utf8_stdout():
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except AttributeError:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, 
                encoding="utf-8", 
                errors="backslashreplace", 
                line_buffering=True
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, 
                encoding="utf-8", 
                errors="backslashreplace", 
                line_buffering=True
            )

    def println(s):
        sys.stdout.write(str(s) + "\n")
        sys.stdout.flush()
    
    def validate_image_path(path):
        return True
    
    def check_image_dimensions(path, max_dim=4096):
        return (0, 0, False)


class GlazeStyleProtector:
    MOMENTUM_BASE_DECAY = 0.9
    MOMENTUM_ADAPTIVE_DECAY = 0.08
    STEP_SIZE_MULTIPLIER = 2.5
    
    LOSS_WEIGHT_TARGET_STYLE = 3.0
    LOSS_WEIGHT_ORIGINAL_STYLE = 2.0
    LOSS_WEIGHT_OTHER_STYLES = 1.0
    
    CLIP_BASE_DIM = 512
    CLIP_LARGE_DIM = 768
    
    STYLE_DESCRIPTIONS = {
        "abstract": [
            "abstract expressionist painting with chaotic brushstrokes",
            "Jackson Pollock style drip painting with random splatter",
            "non-representational abstract art with geometric shapes",
            "pure abstract composition with no recognizable forms"
        ],
        "impressionist": [
            "impressionist painting with visible brushstrokes",
            "Claude Monet style soft focus landscape",
            "post-impressionist artwork with bright colors",
            "impressionist style with dappled light and loose brushwork"
        ],
        "cubist": [
            "cubist painting with geometric fragmentation",
            "Picasso style analytical cubism with multiple perspectives",
            "abstract cubist composition breaking forms into planes",
            "cubist artwork with angular geometric shapes"
        ],
        "sketch": [
            "rough pencil sketch with loose lines",
            "hand-drawn charcoal sketch with hatching",
            "preliminary drawing with construction lines",
            "black and white pencil drawing with shading"
        ],
        "watercolor": [
            "watercolor painting with soft washes",
            "transparent watercolor with flowing pigments",
            "loose watercolor sketch with wet-on-wet technique",
            "delicate watercolor illustration with light colors"
        ]
    }
    
    def __init__(
        self, 
        target_style: str = "abstract", 
        intensity: float = 0.45, 
        iterations: int = 250,
        verbose: bool = True
    ):
        if target_style not in self.STYLE_DESCRIPTIONS:
            raise ValueError(
                f"Unsupported style: {target_style}. "
                f"Choose from: {list(self.STYLE_DESCRIPTIONS.keys())}"
            )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_style = target_style
        self.intensity = intensity
        self.iterations = iterations
        self.verbose = verbose
        
        if verbose:
            println("=== GLAZE-STYLE PROTECTION MODULE ===")
            println(f"STATUS: Device: {self.device}")
            println(f"STATUS: Target Style: {target_style}")
        
        if not CLIP_AVAILABLE:
            raise RuntimeError(
                "CLIP is required! Install with: "
                "pip install git+https://github.com/openai/CLIP.git"
            )
        
        if verbose:
            println("STATUS: Loading CLIP ViT-B/32...")
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        
        if verbose:
            println("STATUS: *** CLIP ViT-B/32 LOADED ***")
        
        try:
            if verbose:
                println("STATUS: Loading CLIP ViT-L/14 (stronger)...")
            self.clip_large, _ = clip.load("ViT-L/14", device=self.device)
            self.clip_large.eval()
            if verbose:
                println("STATUS: *** CLIP ViT-L/14 LOADED ***")
            self.has_large = True
        except Exception:
            if verbose:
                println("WARNING: ViT-L/14 not available, using ViT-B/32 only")
            self.has_large = False
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        if self.has_large:
            for param in self.clip_large.parameters():
                param.requires_grad = False
        
        self.style_embeddings = self._compute_style_embeddings()
        
        if verbose:
            println("STATUS: Style embeddings computed")
    
    def _compute_style_embeddings(self) -> Dict[str, Dict[str, torch.Tensor]]:
        embeddings = {}
        
        with torch.no_grad():
            for style_name, descriptions in self.STYLE_DESCRIPTIONS.items():
                embeddings[style_name] = {}
                
                tokens = clip.tokenize(descriptions).to(self.device)
                
                features_base = self.clip_model.encode_text(tokens)
                features_base = features_base / features_base.norm(dim=-1, keepdim=True)
                avg_embedding_base = features_base.mean(dim=0, keepdim=True)
                avg_embedding_base = avg_embedding_base / avg_embedding_base.norm(dim=-1, keepdim=True)
                embeddings[style_name]['base'] = avg_embedding_base
                
                if self.has_large:
                    features_large = self.clip_large.encode_text(tokens)
                    features_large = features_large / features_large.norm(dim=-1, keepdim=True)
                    avg_embedding_large = features_large.mean(dim=0, keepdim=True)
                    avg_embedding_large = avg_embedding_large / avg_embedding_large.norm(dim=-1, keepdim=True)
                    embeddings[style_name]['large'] = avg_embedding_large
        
        return embeddings
    
    def get_style_embedding(self, style_name: str, model: str = 'base') -> torch.Tensor:
        style_embeddings = self.style_embeddings.get(style_name, {})
        return style_embeddings.get(model)
    
    def style_loss(
        self, 
        img_tensor: torch.Tensor, 
        target_style_embedding: torch.Tensor,
        clip_model
    ) -> torch.Tensor:
        
        resized = F.interpolate(
            img_tensor, 
            size=(224, 224), 
            mode='bicubic', 
            align_corners=False
        )
        
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)
        normalized = (resized - mean) / std
        
        image_features = clip_model.encode_image(normalized)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        sim_to_target = (image_features @ target_style_embedding.T).mean()
        loss1 = -sim_to_target
        
        original_prompts = [
            "a realistic photograph",
            "a clear digital image",
            "photorealistic rendering",
            "high quality photograph"
        ]
        original_tokens = clip.tokenize(original_prompts).to(self.device)
        original_features = clip_model.encode_text(original_tokens)
        original_features = original_features / original_features.norm(dim=-1, keepdim=True)
        sim_to_original = (image_features @ original_features.T).mean()
        loss2 = sim_to_original
        
        model_key = 'large' if image_features.shape[-1] == self.CLIP_LARGE_DIM else 'base'
        
        other_styles_loss = 0
        for style_name, style_dict in self.style_embeddings.items():
            if style_name != self.target_style:
                embedding = style_dict.get(model_key)
                if embedding is not None:
                    sim = (image_features @ embedding.T).mean()
                    other_styles_loss += sim
        loss3 = other_styles_loss / max(1, len(self.style_embeddings) - 1)
        
        total_loss = (self.LOSS_WEIGHT_TARGET_STYLE * loss1 + 
                     self.LOSS_WEIGHT_ORIGINAL_STYLE * loss2 + 
                     self.LOSS_WEIGHT_OTHER_STYLES * loss3)
        
        return total_loss
    
    def multi_model_style_attack(self, img_tensor: torch.Tensor) -> torch.Tensor:
        target_embedding_base = self.get_style_embedding(self.target_style, 'base')
        
        loss = self.style_loss(img_tensor, target_embedding_base, self.clip_model)
        
        if self.has_large:
            target_embedding_large = self.get_style_embedding(self.target_style, 'large')
            loss_large = self.style_loss(img_tensor, target_embedding_large, self.clip_large)
            loss = loss + 1.5 * loss_large
        
        return loss
    
    def apply_style_shift(self, img_tensor: torch.Tensor) -> torch.Tensor:
        perturbed = img_tensor.clone().detach()
        momentum = torch.zeros_like(img_tensor)
        
        if self.verbose:
            println(f"STATUS: Shifting style to '{self.target_style}'...")
            println(f"STATUS: This will take 2-4 minutes...")
        
        best_loss = float('inf')
        best_img = perturbed.clone()
        
        for i in range(self.iterations):
            perturbed.requires_grad = True
            
            loss = self.multi_model_style_attack(perturbed)
            
            if perturbed.grad is not None:
                perturbed.grad.zero_()
            
            loss.backward()
            grad = perturbed.grad
            
            grad_norm = torch.mean(torch.abs(grad))
            if grad_norm > 0:
                grad = grad / grad_norm
            
            decay = self.MOMENTUM_BASE_DECAY + self.MOMENTUM_ADAPTIVE_DECAY * (i / self.iterations)
            momentum = decay * momentum + grad
            
            with torch.no_grad():
                alpha = self.STEP_SIZE_MULTIPLIER * self.intensity / self.iterations
                perturbed = perturbed - alpha * momentum.sign()
                perturbed = torch.clamp(perturbed, 0, 1)
                
                delta = perturbed - img_tensor
                delta = torch.clamp(delta, -self.intensity, self.intensity)
                perturbed = torch.clamp(img_tensor + delta, 0, 1)
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_img = perturbed.clone()
            
            if self.verbose and (i + 1) % 20 == 0:
                println(f"STATUS: Iter {i+1}/{self.iterations} | Loss: {loss.item():.4f}")
        
        if self.verbose:
            println("STATUS: Style shift complete! Using best iteration.")
        return best_img
    
    def protect_image(
        self, 
        input_path: str, 
        output_path: str,
        output_quality: int = 92
    ) -> bool:
        try:
            if self.verbose:
                println("STATUS: Loading image...")
            
            validate_image_path(input_path)
            check_image_dimensions(input_path)
            
            original_img = Image.open(input_path).convert('RGB')
            
            if self.verbose:
                println(f"STATUS: Size: {original_img.size[0]}x{original_img.size[1]}")
            
            img_array = np.array(original_img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            if self.verbose:
                println("STATUS: Applying Glaze-style protection...")
                println(f"STATUS: Intensity={self.intensity}, Iterations={self.iterations}")
            
            protected_tensor = self.apply_style_shift(img_tensor)
            
            protected_array = protected_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            
            diff = np.abs(img_array - protected_array).mean()
            if self.verbose:
                println(f"STATUS: Difference: {diff * 255:.2f}/255")
            
            protected_array = np.clip(protected_array * 255, 0, 255).astype(np.uint8)
            protected_img = Image.fromarray(protected_array, mode='RGB')
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.lower().endswith('.png'):
                protected_img.save(output_path, format='PNG', compress_level=6)
                if self.verbose:
                    println("STATUS: Saved PNG")
            else:
                protected_img.save(
                    output_path, 
                    format='JPEG', 
                    quality=output_quality,
                    subsampling=2,
                    optimize=True
                )
                if self.verbose:
                    println(f"STATUS: Saved JPEG (quality={output_quality}, subsampling=4:2:0)")
            
            if self.verbose:
                output_size = Path(output_path).stat().st_size / 1024
                input_size = Path(input_path).stat().st_size / 1024
                println(f"STATUS: File size: {input_size:.1f}KB → {output_size:.1f}KB")
            
            if self.verbose:
                println("")
                println("=== COMPLETE ===")
                println(f"Image protected with Glaze-style '{self.target_style}' cloaking.")
                println("")
            
            return True
            
        except Exception as e:
            println(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def protect_image(
    input_path: str,
    output_path: str,
    target_style: str = "abstract",
    intensity: float = 0.45,
    iterations: int = 250,
    output_quality: int = 92,
    verbose: bool = True
) -> bool:
    protector = GlazeStyleProtector(
        target_style=target_style,
        intensity=intensity,
        iterations=iterations,
        verbose=verbose
    )
    return protector.protect_image(input_path, output_path, output_quality)
