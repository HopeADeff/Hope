#!/usr/bin/env python3
"""
Example: Image Hashing

This script demonstrates how to use the image hashing module
to compute and compare perceptual hashes for image verification.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_hashing import (
    ImageHasher, 
    compare_hashes, 
    is_similar,
    save_hash_to_file,
    verify_image,
    hamming_distance
)


def example_compute_hash():
    """Example 1: Compute hash for a single image."""
    print("=" * 60)
    print("Example 1: Compute Image Hash")
    print("=" * 60)
    
    hasher = ImageHasher()
    
    # Compute different types of hashes
    image_path = "sample_image.jpg"
    
    print(f"Image: {image_path}\n")
    
    if Path(image_path).exists():
        ahash = hasher.average_hash(image_path)
        dhash = hasher.difference_hash(image_path)
        phash = hasher.perceptual_hash(image_path)
        
        print(f"Average Hash (aHash):    {ahash}")
        print(f"Difference Hash (dHash): {dhash}")
        print(f"Perceptual Hash (pHash): {phash}")
    else:
        print(f"⚠ File not found: {image_path}")
    print()


def example_compare_images():
    """Example 2: Compare two images."""
    print("=" * 60)
    print("Example 2: Compare Two Images")
    print("=" * 60)
    
    hasher = ImageHasher()
    
    image1 = "original.jpg"
    image2 = "modified.jpg"
    
    if Path(image1).exists() and Path(image2).exists():
        # Compute hashes
        hash1 = hasher.perceptual_hash(image1)
        hash2 = hasher.perceptual_hash(image2)
        
        # Compare
        similarity = compare_hashes(hash1, hash2)
        distance = hamming_distance(hash1, hash2)
        
        print(f"Image 1: {image1}")
        print(f"Hash 1:  {hash1}\n")
        print(f"Image 2: {image2}")
        print(f"Hash 2:  {hash2}\n")
        print(f"Similarity: {similarity:.2%}")
        print(f"Hamming Distance: {distance} bits")
        print(f"Similar: {is_similar(hash1, hash2, threshold=0.9)}")
    else:
        print(f"⚠ One or both files not found")
    print()


def example_save_and_verify():
    """Example 3: Save hash and verify later."""
    print("=" * 60)
    print("Example 3: Save and Verify Hash")
    print("=" * 60)
    
    original_image = "protected_image.jpg"
    hash_file = "image_hash.json"
    test_image = "test_image.jpg"
    
    # Save hash
    if Path(original_image).exists():
        save_hash_to_file(original_image, hash_file, method="phash")
        print(f"✓ Hash saved to {hash_file}")
        
        # Later, verify another image
        if Path(test_image).exists():
            is_match, similarity = verify_image(test_image, hash_file, threshold=0.9)
            
            print(f"\nVerifying: {test_image}")
            print(f"Similarity: {similarity:.2%}")
            print(f"Match: {'YES ✓' if is_match else 'NO ✗'}")
        else:
            print(f"⚠ Test file not found: {test_image}")
    else:
        print(f"⚠ Original file not found: {original_image}")
    print()


def example_detect_modifications():
    """Example 4: Detect unauthorized modifications."""
    print("=" * 60)
    print("Example 4: Detect Unauthorized Modifications")
    print("=" * 60)
    
    hasher = ImageHasher()
    
    # Simulate checking multiple versions of an image
    test_cases = [
        ("original.jpg", "Original image"),
        ("slightly_modified.jpg", "Slightly modified (brightness)"),
        ("heavily_modified.jpg", "Heavily modified (cropped)"),
        ("unauthorized_copy.jpg", "Unauthorized copy")
    ]
    
    original_path = "reference_image.jpg"
    
    if Path(original_path).exists():
        original_hash = hasher.perceptual_hash(original_path)
        print(f"Reference: {original_path}")
        print(f"Hash: {original_hash}\n")
        
        for test_path, description in test_cases:
            if Path(test_path).exists():
                test_hash = hasher.perceptual_hash(test_path)
                similarity = compare_hashes(original_hash, test_hash)
                
                status = "✓ MATCH" if similarity >= 0.9 else "✗ DIFFERENT"
                print(f"{description:30} | Similarity: {similarity:.2%} | {status}")
            else:
                print(f"{description:30} | ⚠ File not found")
    else:
        print(f"⚠ Reference file not found: {original_path}")
    print()


def example_batch_hashing():
    """Example 5: Batch hash computation for a directory."""
    print("=" * 60)
    print("Example 5: Batch Hashing")
    print("=" * 60)
    
    hasher = ImageHasher()
    image_dir = Path("images")
    
    if image_dir.exists():
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        print(f"Processing {len(image_files)} images...\n")
        
        hashes = {}
        for img_path in image_files:
            try:
                hash_value = hasher.perceptual_hash(str(img_path))
                hashes[img_path.name] = hash_value
                print(f"✓ {img_path.name:30} | {hash_value}")
            except Exception as e:
                print(f"✗ {img_path.name:30} | Error: {str(e)}")
        
        # Find duplicates
        print("\n" + "-" * 60)
        print("Duplicate Detection:")
        
        checked = set()
        found_duplicates = False
        
        for name1, hash1 in hashes.items():
            for name2, hash2 in hashes.items():
                if name1 != name2 and (name1, name2) not in checked:
                    checked.add((name1, name2))
                    checked.add((name2, name1))
                    
                    if is_similar(hash1, hash2, threshold=0.95):
                        similarity = compare_hashes(hash1, hash2)
                        print(f"  {name1} ≈ {name2} ({similarity:.2%})")
                        found_duplicates = True
        
        if not found_duplicates:
            print("  No duplicates found")
    else:
        print(f"⚠ Directory not found: {image_dir}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("IMAGE HASHING - EXAMPLES")
    print("=" * 60 + "\n")
    
    print("NOTE: This example script demonstrates the API.")
    print("      Actual image files are required to run successfully.")
    print()
    
    # Uncomment the examples you want to run:
    
    # example_compute_hash()
    # example_compare_images()
    # example_save_and_verify()
    # example_detect_modifications()
    # example_batch_hashing()
    
    print("\nQuick Start:")
    print("-" * 60)
    print("from image_hashing import ImageHasher, compare_hashes")
    print()
    print("hasher = ImageHasher()")
    print("hash1 = hasher.perceptual_hash('image1.jpg')")
    print("hash2 = hasher.perceptual_hash('image2.jpg')")
    print("similarity = compare_hashes(hash1, hash2)")
    print("print(f'Similarity: {similarity:.2%}')")
    print("-" * 60)


if __name__ == "__main__":
    main()
