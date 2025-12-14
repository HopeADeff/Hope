#!/usr/bin/env python3
"""
Robust Watermarking Module for Hope-AD
Implements DWT and FFT watermarking that survives compression and geometric transformations.
"""

import sys
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Union, Dict
import hashlib

try:
    from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

try:
    from utils import println, ensure_utf8_stdout
except ImportError:
    def println(s):
        sys.stdout.write(str(s) + "\n")
        sys.stdout.flush()
    def ensure_utf8_stdout():
        pass


class DWTWatermark:
    """DWT based watermarking. Robust against JPEG compression."""
    
    def __init__(self, wavelet: str = 'haar', level: int = 2, 
                 strength: float = 0.1, verbose: bool = True):
        if not PYWT_AVAILABLE:
            raise RuntimeError("PyWavelets required. Install: pip install PyWavelets")
        
        self.wavelet = wavelet
        self.level = level
        self.strength = strength
        self.verbose = verbose
    
    def _generate_watermark_pattern(self, shape: Tuple[int, int], key: str) -> np.ndarray:
        seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        pattern = rng.randn(*shape)
        pattern = pattern / np.abs(pattern).max()
        return pattern
    
    def embed(self, image: Image.Image, key: str, message: str = "") -> Image.Image:
        img_array = np.array(image.convert('RGB'), dtype=np.float64)
        
        for c in range(3):
            channel = img_array[:,:,c]
            coeffs = pywt.wavedec2(channel, self.wavelet, level=self.level)
            cA = coeffs[0]
            pattern = self._generate_watermark_pattern(cA.shape, key + str(c))
            coeffs[0] = cA + self.strength * pattern * np.std(cA)
            
            for i in range(1, min(len(coeffs), 3)):
                cH, cV, cD = coeffs[i]
                pattern_h = self._generate_watermark_pattern(cH.shape, key + f"H{c}{i}")
                pattern_v = self._generate_watermark_pattern(cV.shape, key + f"V{c}{i}")
                coeffs[i] = (
                    cH + self.strength * 0.5 * pattern_h * np.std(cH),
                    cV + self.strength * 0.5 * pattern_v * np.std(cV),
                    cD
                )
            
            img_array[:,:,c] = pywt.waverec2(coeffs, self.wavelet)[:channel.shape[0], :channel.shape[1]]
        
        img_array = np.clip(img_array, 0, 255)
        
        if self.verbose:
            println(f"DWT watermark embedded (key hash: {hashlib.sha256(key.encode()).hexdigest()[:8]})")
        
        return Image.fromarray(img_array.astype(np.uint8), mode='RGB')
    
    def detect(self, image: Image.Image, key: str, threshold: float = 0.3) -> Tuple[bool, float]:
        img_array = np.array(image.convert('RGB'), dtype=np.float64)
        correlations = []
        
        for c in range(3):
            channel = img_array[:,:,c]
            coeffs = pywt.wavedec2(channel, self.wavelet, level=self.level)
            cA = coeffs[0]
            pattern = self._generate_watermark_pattern(cA.shape, key + str(c))
            correlation = np.corrcoef(cA.flatten(), pattern.flatten())[0, 1]
            if not np.isnan(correlation):
                correlations.append(abs(correlation))
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        detected = avg_correlation > threshold
        
        if self.verbose:
            println(f"DWT detection: correlation={avg_correlation:.4f}, detected={detected}")
        
        return detected, avg_correlation


class FFTWatermark:
    """FFT based watermarking. Robust against geometric transformations."""
    
    def __init__(self, strength: float = 50.0, radius: float = 0.3, verbose: bool = True):
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy required for FFT watermarking")
        
        self.strength = strength
        self.radius = radius
        self.verbose = verbose
    
    def _generate_watermark_ring(self, shape: Tuple[int, int], key: str) -> np.ndarray:
        h, w = shape
        center_h, center_w = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((y - center_h)**2 + (x - center_w)**2)
        inner_radius = min(h, w) * self.radius * 0.8
        outer_radius = min(h, w) * self.radius * 1.2
        ring_mask = (dist >= inner_radius) & (dist <= outer_radius)
        
        seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        pattern = np.zeros(shape)
        pattern[ring_mask] = rng.choice([-1, 1], size=ring_mask.sum())
        return pattern
    
    def embed(self, image: Image.Image, key: str) -> Image.Image:
        img_array = np.array(image.convert('RGB'), dtype=np.float64)
        
        for c in range(3):
            channel = img_array[:,:,c]
            f_transform = fft2(channel)
            f_shift = fftshift(f_transform)
            pattern = self._generate_watermark_ring(channel.shape, key + str(c))
            magnitude = np.abs(f_shift)
            phase = np.angle(f_shift)
            magnitude = magnitude + self.strength * pattern * np.mean(magnitude)
            f_shift = magnitude * np.exp(1j * phase)
            f_transform = ifftshift(f_shift)
            channel = np.real(ifft2(f_transform))
            img_array[:,:,c] = channel
        
        img_array = np.clip(img_array, 0, 255)
        
        if self.verbose:
            println(f"FFT watermark embedded (key hash: {hashlib.sha256(key.encode()).hexdigest()[:8]})")
        
        return Image.fromarray(img_array.astype(np.uint8), mode='RGB')
    
    def detect(self, image: Image.Image, key: str, threshold: float = 0.2) -> Tuple[bool, float]:
        img_array = np.array(image.convert('RGB'), dtype=np.float64)
        correlations = []
        
        for c in range(3):
            channel = img_array[:,:,c]
            f_transform = fft2(channel)
            f_shift = fftshift(f_transform)
            magnitude = np.abs(f_shift)
            pattern = self._generate_watermark_ring(channel.shape, key + str(c))
            ring_mask = pattern != 0
            
            if ring_mask.sum() > 0:
                extracted = magnitude[ring_mask]
                expected = pattern[ring_mask]
                correlation = np.corrcoef(extracted, expected)[0, 1]
                if not np.isnan(correlation):
                    correlations.append(abs(correlation))
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        detected = avg_correlation > threshold
        
        if self.verbose:
            println(f"FFT detection: correlation={avg_correlation:.4f}, detected={detected}")
        
        return detected, avg_correlation


class RobustWatermarker:
    """Combined robust watermarking using DWT and FFT."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.dwt = DWTWatermark(verbose=False) if PYWT_AVAILABLE else None
        self.fft = FFTWatermark(verbose=False) if SCIPY_AVAILABLE else None
        
        if not self.dwt and not self.fft:
            raise RuntimeError("PyWavelets or scipy required. Install: pip install PyWavelets scipy")
    
    def embed(self, image: Union[str, Image.Image], key: str,
              use_dwt: bool = True, use_fft: bool = True) -> Image.Image:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        if self.verbose:
            println("Applying robust watermark...")
        
        result = image
        
        if use_dwt and self.dwt:
            result = self.dwt.embed(result, key + "_dwt")
        
        if use_fft and self.fft:
            result = self.fft.embed(result, key + "_fft")
        
        if self.verbose:
            methods = []
            if use_dwt and self.dwt:
                methods.append("DWT")
            if use_fft and self.fft:
                methods.append("FFT")
            println(f"Robust watermark complete: {' + '.join(methods)}")
        
        return result
    
    def detect(self, image: Union[str, Image.Image], key: str,
               threshold: float = 0.25) -> Tuple[bool, Dict]:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        details = {}
        detected_any = False
        
        if self.dwt:
            dwt_detected, dwt_corr = self.dwt.detect(image, key + "_dwt", threshold)
            details['dwt'] = {'detected': dwt_detected, 'correlation': dwt_corr}
            if dwt_detected:
                detected_any = True
        
        if self.fft:
            fft_detected, fft_corr = self.fft.detect(image, key + "_fft", threshold)
            details['fft'] = {'detected': fft_detected, 'correlation': fft_corr}
            if fft_detected:
                detected_any = True
        
        if self.verbose:
            for method, data in details.items():
                status = "FOUND" if data['detected'] else "not found"
                println(f"{method.upper()}: {status} (corr={data['correlation']:.4f})")
        
        return detected_any, details
    
    def embed_to_file(self, input_path: str, output_path: str, 
                      key: str, quality: int = 95) -> bool:
        try:
            image = Image.open(input_path).convert('RGB')
            result = self.embed(image, key)
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.lower().endswith('.png'):
                result.save(output_path, format='PNG')
            else:
                result.save(output_path, format='JPEG', quality=quality)
            
            if self.verbose:
                input_size = Path(input_path).stat().st_size / 1024
                output_size = Path(output_path).stat().st_size / 1024
                println(f"Saved: {output_path} ({input_size:.1f}KB → {output_size:.1f}KB)")
            
            return True
            
        except Exception as e:
            println(f"Error: {e}")
            return False


def embed_robust_watermark(input_path: str, output_path: str, 
                           key: str, verbose: bool = True) -> bool:
    watermarker = RobustWatermarker(verbose=verbose)
    return watermarker.embed_to_file(input_path, output_path, key)


def detect_watermark(image_path: str, key: str) -> Tuple[bool, Dict]:
    watermarker = RobustWatermarker(verbose=False)
    return watermarker.detect(image_path, key)


if __name__ == "__main__":
    ensure_utf8_stdout()
    print("Robust Watermarking Module")
    print("Usage:")
    print("  from robust_watermark import embed_robust_watermark, detect_watermark")
    print("  embed_robust_watermark('input.jpg', 'output.jpg', 'secret_key')")
