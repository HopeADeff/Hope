#!/usr/bin/env python3
"""
Nightshade Protection Module for Hope-AD
Implements concept poisoning to protect images from AI training.

Based on research: "Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models"
Shan et al., arXiv:2310.13828
"""

import sys
import io
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict, List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    from utils import println, ensure_utf8_stdout, validate_image_path
    from gpu_utils import get_device, is_cuda_available
except ImportError:
    def println(s):
        sys.stdout.write(str(s) + "\n")
        sys.stdout.flush()
    def ensure_utf8_stdout():
        pass
    def validate_image_path(p):
        return True
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
    def is_cuda_available():
        return TORCH_AVAILABLE and torch.cuda.is_available()


class NightshadeProtector:
    """
    Nightshade-style concept poisoning protection.
    
    This modifies images so that when AI models train on them,
    they learn incorrect associations (e.g., "dog" → looks like "cat").
    """
    
    CONCEPT_PAIRS = {
        "dog": "cat",
        "cat": "dog", 
        "car": "bicycle",
        "person": "mannequin",
        "landscape": "abstract",
        "portrait": "sculpture",
        "building": "ruins",
        "food": "plastic",
        "animal": "robot",
        "nature": "synthetic"
    }
    
    def __init__(
        self,
        source_concept: str = "artwork",
        target_concept: str = "noise",
        intensity: float = 0.08,
        iterations: int = 200,
        verbose: bool = True
    ):
        """
        Initialize Nightshade protector.
        
        Args:
            source_concept: What the image actually is
            target_concept: What AI should learn it as (poison)
            intensity: Perturbation strength (0.01-0.15)
            iterations: Optimization steps
            verbose: Print progress
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for Nightshade protection")
        
        if not CLIP_AVAILABLE:
            raise RuntimeError(
                "CLIP is required for Nightshade protection. "
                "Install with: pip install git+https://github.com/openai/CLIP.git"
            )
        
        self.device = get_device()
        self.source_concept = source_concept
        self.target_concept = target_concept
        self.intensity = intensity
        self.iterations = iterations
        self.verbose = verbose
        
        if verbose:
            println("=== NIGHTSHADE PROTECTION MODULE ===")
            println(f"Device: {self.device}")
            println(f"Poisoning: '{source_concept}' → '{target_concept}'")
        
        if verbose:
            println("STATUS: Loading CLIP model...")
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self._compute_concept_embeddings()
        
        if verbose:
            println("STATUS: Nightshade ready")
    
    def _compute_concept_embeddings(self):
        """Pre-compute text embeddings for concepts."""
        source_prompts = [
            f"a {self.source_concept}",
            f"an image of {self.source_concept}",
            f"a photo of {self.source_concept}",
            f"{self.source_concept} artwork"
        ]
        
        target_prompts = [
            f"a {self.target_concept}",
            f"an image of {self.target_concept}",
            f"a photo of {self.target_concept}",
            f"{self.target_concept} texture"
        ]
        
        with torch.no_grad():
            source_tokens = clip.tokenize(source_prompts).to(self.device)
            source_features = self.clip_model.encode_text(source_tokens)
            source_features = source_features / source_features.norm(dim=-1, keepdim=True)
            self.source_embedding = source_features.mean(dim=0, keepdim=True)
            self.source_embedding = self.source_embedding / self.source_embedding.norm(dim=-1, keepdim=True)
            
            target_tokens = clip.tokenize(target_prompts).to(self.device)
            target_features = self.clip_model.encode_text(target_tokens)
            target_features = target_features / target_features.norm(dim=-1, keepdim=True)
            self.target_embedding = target_features.mean(dim=0, keepdim=True)
            self.target_embedding = self.target_embedding / self.target_embedding.norm(dim=-1, keepdim=True)
    
    def _get_image_embedding(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Get CLIP embedding for image tensor."""
        resized = F.interpolate(img_tensor, size=(224, 224), mode='bicubic', align_corners=False)
        
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)
        normalized = (resized - mean) / std
        
        features = self.clip_model.encode_image(normalized)
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features
    
    def _concept_poison_loss(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss for concept poisoning.
        
        Goal: Make image embedding closer to target concept,
        while staying away from source concept.
        """
        img_features = self._get_image_embedding(img_tensor)
        
        sim_to_target = (img_features @ self.target_embedding.T).mean()
        loss_target = -sim_to_target * 2.0
        
        sim_to_source = (img_features @ self.source_embedding.T).mean()
        loss_source = sim_to_source * 1.5
        
        feature_entropy = -torch.sum(F.softmax(img_features, dim=-1) * 
                                     F.log_softmax(img_features, dim=-1), dim=-1).mean()
        loss_entropy = -feature_entropy * 0.1
        
        total_loss = loss_target + loss_source + loss_entropy
        
        return total_loss
    
    def apply_poison(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Apply concept poisoning to image tensor."""
        perturbed = img_tensor.clone().detach()
        momentum = torch.zeros_like(img_tensor)
        
        if self.verbose:
            println(f"STATUS: Applying concept poison ({self.iterations} iterations)...")
            println(f"STATUS: '{self.source_concept}' → '{self.target_concept}'")
        
        best_loss = float('inf')
        best_img = perturbed.clone()
        
        for i in range(self.iterations):
            perturbed.requires_grad = True
            
            loss = self._concept_poison_loss(perturbed)
            
            if perturbed.grad is not None:
                perturbed.grad.zero_()
            
            loss.backward()
            grad = perturbed.grad
            
            grad_norm = torch.mean(torch.abs(grad))
            if grad_norm > 0:
                grad = grad / grad_norm
            
            decay = 0.9 + 0.05 * (i / self.iterations)
            momentum = decay * momentum + grad
            
            with torch.no_grad():
                alpha = 2.5 * self.intensity / self.iterations
                perturbed = perturbed - alpha * momentum.sign()
                perturbed = torch.clamp(perturbed, 0, 1)
                
                delta = perturbed - img_tensor
                delta = torch.clamp(delta, -self.intensity, self.intensity)
                perturbed = torch.clamp(img_tensor + delta, 0, 1)
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_img = perturbed.clone()
            
            if self.verbose and (i + 1) % 25 == 0:
                println(f"STATUS: Iter {i+1}/{self.iterations} | Loss: {loss.item():.4f}")
        
        if self.verbose:
            println("STATUS: Concept poisoning complete")
        
        return best_img
    
    def protect_image(
        self,
        input_path: str,
        output_path: str,
        output_quality: int = 92
    ) -> bool:
        """
        Apply Nightshade protection to an image.
        
        Args:
            input_path: Path to input image
            output_path: Path to save protected image
            output_quality: JPEG quality (85-98)
            
        Returns:
            True if successful
        """
        try:
            if self.verbose:
                println("STATUS: Loading image...")
            
            validate_image_path(input_path)
            
            original_img = Image.open(input_path).convert('RGB')
            
            if self.verbose:
                println(f"STATUS: Size: {original_img.size[0]}x{original_img.size[1]}")
            
            img_array = np.array(original_img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            protected_tensor = self.apply_poison(img_tensor)
            
            protected_array = protected_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            
            diff = np.abs(img_array - protected_array).mean()
            if self.verbose:
                println(f"STATUS: Avg difference: {diff * 255:.2f}/255")
            
            protected_array = np.clip(protected_array * 255, 0, 255).astype(np.uint8)
            protected_img = Image.fromarray(protected_array, mode='RGB')
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.lower().endswith('.png'):
                protected_img.save(output_path, format='PNG', compress_level=6)
            else:
                protected_img.save(
                    output_path,
                    format='JPEG',
                    quality=output_quality,
                    subsampling=2,
                    optimize=True
                )
            
            if self.verbose:
                input_size = Path(input_path).stat().st_size / 1024
                output_size = Path(output_path).stat().st_size / 1024
                println(f"STATUS: File size: {input_size:.1f}KB → {output_size:.1f}KB")
                println("")
                println("=== NIGHTSHADE PROTECTION COMPLETE ===")
                println(f"Concept poisoned: '{self.source_concept}' → '{self.target_concept}'")
            
            return True
            
        except Exception as e:
            println(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def protect_image(
    input_path: str,
    output_path: str,
    source_concept: str = "artwork",
    target_concept: str = "noise",
    intensity: float = 0.08,
    iterations: int = 200,
    output_quality: int = 92,
    verbose: bool = True
) -> bool:
    """
    Convenience function for Nightshade protection.
    
    Args:
        input_path: Input image path
        output_path: Output image path
        source_concept: What the image is
        target_concept: What AI should wrongly learn
        intensity: Protection strength
        iterations: Optimization iterations
        output_quality: JPEG quality
        verbose: Print progress
        
    Returns:
        True if successful
    """
    protector = NightshadeProtector(
        source_concept=source_concept,
        target_concept=target_concept,
        intensity=intensity,
        iterations=iterations,
        verbose=verbose
    )
    return protector.protect_image(input_path, output_path, output_quality)


if __name__ == "__main__":
    ensure_utf8_stdout()
    
    print("Nightshade Protection Module")
    print("=" * 40)
    print("\nUsage:")
    print("  from nightshade_protection import protect_image")
    print("  protect_image('input.jpg', 'output.jpg', source_concept='dog', target_concept='cat')")
    print("\nAvailable concept pairs:")
    for source, target in NightshadeProtector.CONCEPT_PAIRS.items():
        print(f"  {source} → {target}")
