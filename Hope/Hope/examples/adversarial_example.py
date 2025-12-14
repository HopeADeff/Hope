#!/usr/bin/env python3
"""
Example: Adversarial Perturbations

This script demonstrates how to use the adversarial perturbations module
to protect images from AI detection.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adversarial_perturbations import AdversarialProtector, protect_image


def example_basic_protection():
    """Example 1: Basic image protection with default settings."""
    print("=" * 60)
    print("Example 1: Basic Image Protection")
    print("=" * 60)
    
    # Using convenience function
    success = protect_image(
        input_path="input_image.jpg",
        output_path="protected_image.jpg"
    )
    
    if success:
        print("✓ Image successfully protected!")
    else:
        print("✗ Protection failed!")
    print()


def example_custom_settings():
    """Example 2: Protection with custom intensity and iterations."""
    print("=" * 60)
    print("Example 2: Custom Settings")
    print("=" * 60)
    
    # Using class directly for more control
    protector = AdversarialProtector(
        intensity=0.25,      # Lower intensity for subtle changes
        iterations=100,      # Fewer iterations for faster processing
        verbose=True
    )
    
    success = protector.protect_image(
        input_path="input_image.jpg",
        output_path="protected_subtle.jpg"
    )
    
    if success:
        print("✓ Image protected with custom settings!")
    else:
        print("✗ Protection failed!")
    print()


def example_batch_processing():
    """Example 3: Batch processing multiple images."""
    print("=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)
    
    # Initialize once for efficiency
    protector = AdversarialProtector(
        intensity=0.30,
        iterations=150,
        verbose=False  # Disable verbose for batch processing
    )
    
    # Process multiple images
    images = ["image1.jpg", "image2.jpg", "image3.jpg"]
    
    for img_path in images:
        if Path(img_path).exists():
            output_path = f"protected_{img_path}"
            print(f"Processing {img_path}...")
            
            success = protector.protect_image(img_path, output_path)
            
            if success:
                print(f"  ✓ Saved to {output_path}")
            else:
                print(f"  ✗ Failed to process {img_path}")
        else:
            print(f"  ⚠ File not found: {img_path}")
    print()


def example_strong_protection():
    """Example 4: Maximum protection with strong settings."""
    print("=" * 60)
    print("Example 4: Strong Protection")
    print("=" * 60)
    
    protector = AdversarialProtector(
        intensity=0.40,      # Higher intensity
        iterations=200,      # More iterations
        verbose=True
    )
    
    success = protector.protect_image(
        input_path="sensitive_image.jpg",
        output_path="highly_protected.jpg"
    )
    
    if success:
        print("✓ Image protected with maximum settings!")
    else:
        print("✗ Protection failed!")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ADVERSARIAL PERTURBATIONS - EXAMPLES")
    print("=" * 60 + "\n")
    
    print("NOTE: This example script demonstrates the API.")
    print("      Actual image files are required to run successfully.")
    print()
    
    # Uncomment the examples you want to run:
    
    # example_basic_protection()
    # example_custom_settings()
    # example_batch_processing()
    # example_strong_protection()
    
    print("\nTo use this module in your code:")
    print("-" * 60)
    print("from adversarial_perturbations import protect_image")
    print()
    print("protect_image(")
    print("    input_path='input.jpg',")
    print("    output_path='output.jpg',")
    print("    intensity=0.30,")
    print("    iterations=150")
    print(")")
    print("-" * 60)


if __name__ == "__main__":
    main()
