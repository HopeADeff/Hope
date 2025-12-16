import sys
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

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


def save_optimized(
    image: Image.Image,
    output_path: str,
    quality: int = 92,
    max_size_kb: Optional[int] = None,
    verbose: bool = True
) -> int:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.lower().endswith('.png'):
        image.save(output_path, format='PNG', compress_level=6)
        size_kb = Path(output_path).stat().st_size / 1024
        if verbose:
            println(f"STATUS: Saved PNG ({size_kb:.1f}KB)")
        return int(size_kb)
    
    current_quality = quality
    
    if max_size_kb is None:
        image.save(
            output_path,
            format='JPEG',
            quality=current_quality,
            subsampling=2,
            optimize=True
        )
        size_kb = Path(output_path).stat().st_size / 1024
        if verbose:
            println(f"STATUS: Saved JPEG (quality={current_quality}, {size_kb:.1f}KB)")
        return int(size_kb)
    
    if verbose:
        println(f"STATUS: Optimizing to max {max_size_kb}KB...")
    
    for attempt_quality in range(current_quality, 84, -2):
        image.save(
            output_path,
            format='JPEG',
            quality=attempt_quality,
            subsampling=2,
            optimize=True
        )
        size_kb = Path(output_path).stat().st_size / 1024
        
        if size_kb <= max_size_kb:
            if verbose:
                println(f"STATUS: Saved JPEG (quality={attempt_quality}, {size_kb:.1f}KB)")
            return int(size_kb)
    
    image.save(
        output_path,
        format='JPEG',
        quality=85,
        subsampling=2,
        optimize=True
    )
    size_kb = Path(output_path).stat().st_size / 1024
    if verbose:
        println(f"STATUS: Saved JPEG (quality=85, {size_kb:.1f}KB) - minimum quality")
    return int(size_kb)


class AdversarialProtector:
    def __init__(self, intensity: float = 0.30, iterations: int = 150, verbose: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.intensity = intensity
        self.iterations = iterations
        self.verbose = verbose
        
        if verbose:
            println("=== ADVERSARIAL PERTURBATIONS MODULE ===")
            println(f"STATUS: Device: {self.device}")
        
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
            println("STATUS: *** CLIP LOADED ***")
        
        try:
            if verbose:
                println("STATUS: Loading CLIP ViT-L/14 (stronger)...")
            self.clip_large, _ = clip.load("ViT-L/14", device=self.device)
            self.clip_large.eval()
            if verbose:
                println("STATUS: *** CLIP LARGE LOADED ***")
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
    
    def get_chaos_prompts(self) -> list:
        return [
            "completely destroyed corrupted visual data",
            "incomprehensible chaotic noise patterns",
            "severely distorted unrecognizable imagery",
            "broken fragmented visual information",
            "extreme digital corruption and artifacts",
            "meaningless random pixel arrangements",
            "visual chaos with no coherent structure",
            "utterly corrupted incomprehensible forms"
        ]
    
    def semantic_loss(self, img_tensor: torch.Tensor, clip_model) -> torch.Tensor:
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
        
        chaos_prompts = self.get_chaos_prompts()
        text_tokens = clip.tokenize(chaos_prompts).to(self.device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        sim_to_chaos = (image_features @ text_features.T).mean()
        loss1 = -sim_to_chaos
        
        feature_std = torch.std(image_features)
        loss2 = -feature_std
        
        normal_prompts = [
            "a clear image", 
            "a recognizable picture", 
            "coherent visual content"
        ]
        normal_tokens = clip.tokenize(normal_prompts).to(self.device)
        normal_features = clip_model.encode_text(normal_tokens)
        normal_features = normal_features / normal_features.norm(dim=-1, keepdim=True)
        sim_to_normal = (image_features @ normal_features.T).mean()
        loss3 = sim_to_normal
        
        total_loss = loss1 + 0.5 * loss2 + 2.0 * loss3
        
        return total_loss
    
    def multi_scale_attack(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply adversarial attack at multiple scales.
        
        Args:
            img_tensor: Input image tensor
        
        Returns:
            Combined multi-scale loss
        """
        scales = [1.0, 0.75, 0.5]
        total_loss = 0
        
        for scale in scales:
            if scale != 1.0:
                h, w = img_tensor.shape[2:]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = F.interpolate(
                    img_tensor, 
                    size=(new_h, new_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                scaled = F.interpolate(
                    scaled, 
                    size=(h, w), 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                scaled = img_tensor
            
            loss = self.semantic_loss(scaled, self.clip_model)
            
            if self.has_large:
                loss_large = self.semantic_loss(scaled, self.clip_large)
                loss = loss + 1.5 * loss_large
            
            total_loss += loss
        
        return total_loss / len(scales)
    
    def add_imperceptible_chaos(self, img_array: np.ndarray) -> np.ndarray:
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
        
        edge1 = img_pil.filter(ImageFilter.EDGE_ENHANCE_MORE)
        edge2 = img_pil.filter(ImageFilter.FIND_EDGES)
        
        enhancer = ImageEnhance.Color(img_pil)
        color_shifted = enhancer.enhance(1.15)
        
        sharp_enhancer = ImageEnhance.Sharpness(img_pil)
        sharpened = sharp_enhancer.enhance(2.0)
        
        img_arr = np.array(img_pil, dtype=np.float32) / 255.0
        edge1_arr = np.array(edge1, dtype=np.float32) / 255.0
        edge2_arr = np.array(edge2, dtype=np.float32) / 255.0
        color_arr = np.array(color_shifted, dtype=np.float32) / 255.0
        sharp_arr = np.array(sharpened, dtype=np.float32) / 255.0
        
        result = (
            img_arr * 0.70 + 
            edge1_arr * 0.08 + 
            edge2_arr * 0.05 +
            color_arr * 0.10 +
            sharp_arr * 0.07
        )
        
        noise = np.random.randn(*result.shape) * 0.015
        result = result + noise
        
        return np.clip(result, 0, 1)
    
    def apply_pgd_attack(self, img_tensor: torch.Tensor) -> torch.Tensor:
        perturbed = img_tensor.clone().detach()
        momentum = torch.zeros_like(img_tensor)
        
        if self.verbose:
            println("STATUS: Starting adversarial attack...")
            println("STATUS: This will take 2-3 minutes...")
        
        best_loss = float('inf')
        best_img = perturbed.clone()
        
        for i in range(self.iterations):
            perturbed.requires_grad = True
            
            loss = self.multi_scale_attack(perturbed)
            
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
            
            if self.verbose and (i + 1) % 15 == 0:
                println(f"STATUS: Iter {i+1}/{self.iterations} | Loss: {loss.item():.4f}")
        
        if self.verbose:
            println("STATUS: Attack complete! Using best iteration.")
        return best_img
    
    def protect_image(
        self, 
        input_path: str, 
        output_path: str, 
        target_description: str = "",
        output_quality: int = 92,
        max_file_size_kb: Optional[int] = None
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
                println("STATUS: Applying adversarial perturbations...")
                println(f"STATUS: Intensity={self.intensity}, Iterations={self.iterations}")
            
            perturbed = self.apply_pgd_attack(img_tensor)
            
            protected_array = perturbed.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            
            if self.verbose:
                println("STATUS: Adding imperceptible chaos layers...")
            protected_array = self.add_imperceptible_chaos(protected_array)
            
            diff = np.abs(img_array - protected_array).mean()
            if self.verbose:
                println(f"STATUS: Difference: {diff * 255:.2f}/255")
            
            protected_array = np.clip(protected_array * 255, 0, 255).astype(np.uint8)
            protected_img = Image.fromarray(protected_array, mode='RGB')
            
            input_size_kb = Path(input_path).stat().st_size / 1024
            
            output_size_kb = save_optimized(
                protected_img,
                output_path,
                quality=output_quality,
                max_size_kb=max_file_size_kb,
                verbose=self.verbose
            )
            
            if self.verbose:
                println(f"STATUS: File size: {input_size_kb:.1f}KB → {output_size_kb}KB")
            
            if self.verbose:
                println("")
                println("=== COMPLETE ===")
                println("Image protected with adversarial perturbations.")
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
    intensity: float = 0.30,
    iterations: int = 150,
    output_quality: int = 92,
    max_file_size_kb: Optional[int] = None,
    verbose: bool = True
) -> bool:
    protector = AdversarialProtector(intensity=intensity, iterations=iterations, verbose=verbose)
    return protector.protect_image(input_path, output_path, output_quality=output_quality, max_file_size_kb=max_file_size_kb)
