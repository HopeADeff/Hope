
import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

def println(s):
    print(s)
    sys.stdout.flush()

def main():
    println("=== Hope-AD Model Downloader ===")
    println("Preparing to download Stable Diffusion v1.5 models...")
    
    base_dir = Path(__file__).parent.parent.parent
    assets_dir = base_dir / "assets" / "models" / "sd-v1-5"
    
    if assets_dir.exists():
        println(f"Warning: Directory already exists at {assets_dir}")
        println("Checking if models are valid...")
        if (assets_dir / "model_index.json").exists() and \
           (assets_dir / "unet" / "diffusion_pytorch_model.bin").exists():
            println("Models seem to be present. Skipping download.")
            return
        else:
            println("Directory exists but models are missing. Redownloading...")
    
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    println(f"Target directory: {assets_dir}")
    println("Downloading from Hugging Face (runwayml/stable-diffusion-v1-5)...")
    println("This is ~4GB of data. Please wait...")
    
    try:
        snapshot_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            local_dir=str(assets_dir),
            ignore_patterns=["*.ckpt", "*.h5", "*.msgpack"], 
            allow_patterns=[
                "feature_extractor/*.json",
                "scheduler/*.json",
                "text_encoder/*.json",
                "text_encoder/model.safetensors",
                "tokenizer/*",
                "unet/*.json",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/*.json",
                "vae/diffusion_pytorch_model.safetensors",
                "model_index.json"
            ],
            local_dir_use_symlinks=False
        )
        
        println("\nSUCCESS: Models downloaded successfully!")
        println(f"Models saved to: {assets_dir}")
        
    except Exception as e:
        println(f"\nERROR: Failed to download models: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
