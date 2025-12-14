#!/usr/bin/env python3
"""
Shared utility functions for Hope-AD image protection modules.
"""

import sys
import io


def ensure_utf8_stdout():
    """Configure stdout and stderr to use UTF-8 encoding."""
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
    """Print a line to stdout with immediate flush."""
    sys.stdout.write(str(s) + "\n")
    sys.stdout.flush()


def validate_image_path(image_path: str) -> bool:
    """
    Validate that the given path exists and appears to be an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    from pathlib import Path
    
    path = Path(image_path)
    
    if not path.exists():
        raise ValueError(f"File not found: {image_path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {image_path}")
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    if path.suffix.lower() not in valid_extensions:
        raise ValueError(
            f"Unsupported image format: {path.suffix}. "
            f"Supported formats: {', '.join(valid_extensions)}"
        )
    
    return True


def get_file_size_kb(file_path: str) -> float:
    """Get file size in kilobytes."""
    from pathlib import Path
    return Path(file_path).stat().st_size / 1024


def check_image_dimensions(image_path: str, max_dimension: int = 4096) -> tuple:
    """
    Check image dimensions and warn if very large.
    
    Args:
        image_path: Path to the image
        max_dimension: Maximum recommended dimension
        
    Returns:
        Tuple of (width, height, is_large)
    """
    from PIL import Image
    
    with Image.open(image_path) as img:
        width, height = img.size
        is_large = width > max_dimension or height > max_dimension
        
        if is_large:
            println(f"WARNING: Large image detected ({width}x{height}). "
                   f"Processing may be slow and use significant memory.")
        
        return width, height, is_large
