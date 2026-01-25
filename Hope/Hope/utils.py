import sys
import io
import os
from pathlib import Path


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


def validate_image_path(image_path: str) -> bool:
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
    return Path(file_path).stat().st_size / 1024


def check_image_dimensions(image_path: str, max_dimension: int = 4096) -> tuple:
    from PIL import Image

    with Image.open(image_path) as img:
        width, height = img.size
        is_large = width > max_dimension or height > max_dimension

        if is_large:
            println(f"WARNING: Large image detected ({width}x{height}). "
                    f"Processing may be slow and use significant memory.")

        return width, height, is_large


def get_model_path(model_name: str) -> str:
    if getattr(sys, 'frozen', False):
        base_path = Path(sys.executable).parent
    else:
        base_path = Path(__file__).parent.parent.parent

    model_path = base_path / "assets" / "models" / model_name

    return str(model_path)
