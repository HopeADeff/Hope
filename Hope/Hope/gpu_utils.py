#!/usr/bin/env python3
"""
GPU Utilities Module for Hope-AD
Provides CUDA detection, memory management, and device optimization.
"""

import sys
import os
from typing import Optional, Tuple, Dict, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GPUManager:
    """Manages GPU resources and provides utilities for CUDA operations."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._device = None
        self._gpu_info = None
        self._initialize()
    
    def _initialize(self):
        """Initialize GPU detection and gather device information."""
        if not TORCH_AVAILABLE:
            if self.verbose:
                print("WARNING: PyTorch not available. GPU features disabled.")
            return
        
        self._gpu_info = self.get_gpu_info()
        self._device = self.get_optimal_device()
        
        if self.verbose:
            self._print_device_info()
    
    def _print_device_info(self):
        """Print device information."""
        if self._gpu_info["cuda_available"]:
            print(f"GPU: {self._gpu_info['gpu_name']}")
            print(f"VRAM: {self._gpu_info['total_memory_gb']:.1f} GB")
            print(f"CUDA: {self._gpu_info['cuda_version']}")
        else:
            print("GPU: Not available (using CPU)")
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """
        Get comprehensive GPU information.
        
        Returns:
            Dictionary with GPU details
        """
        info = {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_name": "N/A",
            "total_memory_gb": 0,
            "free_memory_gb": 0,
            "cuda_version": "N/A",
            "cudnn_version": "N/A",
            "torch_version": "N/A"
        }
        
        if not TORCH_AVAILABLE:
            return info
        
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        
        if info["cuda_available"]:
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda or "N/A"
            
            total_mem = torch.cuda.get_device_properties(0).total_memory
            info["total_memory_gb"] = total_mem / (1024 ** 3)
            
            try:
                free_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
                info["free_memory_gb"] = free_mem / (1024 ** 3)
            except:
                info["free_memory_gb"] = info["total_memory_gb"]
            
            if torch.backends.cudnn.is_available():
                info["cudnn_version"] = str(torch.backends.cudnn.version())
        
        return info
    
    @staticmethod
    def get_optimal_device() -> "torch.device":
        """
        Get the optimal device for computation.
        
        Returns:
            torch.device - CUDA if available, else CPU
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")
    
    @property
    def device(self) -> "torch.device":
        """Get the current device."""
        if self._device is None:
            self._device = self.get_optimal_device()
        return self._device
    
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory cache."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def get_memory_usage() -> Tuple[float, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Tuple of (allocated_gb, reserved_gb)
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return (0.0, 0.0)
        
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        return (allocated, reserved)
    
    def check_memory_for_image(self, width: int, height: int, 
                                batch_size: int = 1,
                                safety_factor: float = 2.0) -> bool:
        """
        Check if there's enough GPU memory for processing an image.
        
        Args:
            width: Image width
            height: Image height
            batch_size: Batch size
            safety_factor: Memory safety multiplier
            
        Returns:
            True if enough memory available
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return True  # CPU mode, assume enough RAM
        
        pixel_count = width * height * 3
        bytes_needed = pixel_count * 4 * batch_size * safety_factor
        gb_needed = bytes_needed / (1024 ** 3)
        
        info = self.get_gpu_info()
        available_gb = info["free_memory_gb"]
        
        return available_gb >= gb_needed
    
    def optimize_for_inference(self):
        """Optimize PyTorch settings for inference."""
        if not TORCH_AVAILABLE:
            return
        
        torch.set_grad_enabled(False)
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
    
    def optimize_for_training(self):
        """Optimize PyTorch settings for training/perturbation generation."""
        if not TORCH_AVAILABLE:
            return
        
        torch.set_grad_enabled(True)
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager(verbose: bool = False) -> GPUManager:
    """Get or create the GPU manager singleton."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager(verbose=verbose)
    return _gpu_manager


def get_device() -> "torch.device":
    """Quick helper to get the optimal device."""
    return get_gpu_manager(verbose=False).device


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


def get_device_name() -> str:
    """Get the name of the current device."""
    if not TORCH_AVAILABLE:
        return "CPU (PyTorch not installed)"
    
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def print_gpu_status():
    """Print detailed GPU status."""
    print("=" * 50)
    print("GPU STATUS")
    print("=" * 50)
    
    if not TORCH_AVAILABLE:
        print("PyTorch: Not installed")
        print("GPU: Not available")
        print("=" * 50)
        return
    
    info = GPUManager.get_gpu_info()
    
    print(f"PyTorch: {info['torch_version']}")
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"GPU Count: {info['gpu_count']}")
        print(f"GPU Name: {info['gpu_name']}")
        print(f"Total VRAM: {info['total_memory_gb']:.2f} GB")
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"cuDNN Version: {info['cudnn_version']}")
        
        allocated, reserved = GPUManager.get_memory_usage()
        print(f"Memory Allocated: {allocated:.2f} GB")
        print(f"Memory Reserved: {reserved:.2f} GB")
    else:
        print("GPU: Not available (using CPU fallback)")
    
    print("=" * 50)


if __name__ == "__main__":
    print_gpu_status()
