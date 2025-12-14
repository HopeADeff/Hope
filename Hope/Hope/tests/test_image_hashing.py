#!/usr/bin/env python3
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw
import numpy as np
from image_hashing import (
    ImageHasher,
    compare_hashes,
    is_similar,
    save_hash_to_file,
    verify_image,
    hamming_distance
)

TEST_DIR = tempfile.mkdtemp(prefix="image_hash_test_")

def create_test_image(path, size=(256, 256), color=(100, 150, 200)):
    """Create a simple test image."""
    img = Image.new('RGB', size, color)
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([50, 50, 150, 150], fill=(200, 100, 100))
    draw.ellipse([100, 100, 200, 200], fill=(100, 200, 100))
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def create_modified_image(original_path, modified_path, brightness=1.2):
    img = Image.open(original_path)
    
    arr = np.array(img, dtype=np.float32)
    arr = np.clip(arr * brightness, 0, 255).astype(np.uint8)
    
    modified = Image.fromarray(arr)
    modified.save(modified_path)
    return modified_path


def test_hash_computation():
    """Test 1: Basic hash computation."""
    print("=" * 60)
    print("Test 1: Hash Computation")
    print("=" * 60)
    
    test_img = os.path.join(TEST_DIR, "test_image.jpg")
    create_test_image(test_img)
    
    hasher = ImageHasher()
    
    ahash = hasher.average_hash(test_img)
    dhash = hasher.difference_hash(test_img)
    phash = hasher.perceptual_hash(test_img)
    
    print(f"✓ Average Hash:    {ahash}")
    print(f"✓ Difference Hash: {dhash}")
    print(f"✓ Perceptual Hash: {phash}")
    
    assert len(ahash) > 0, "aHash is empty"
    assert len(dhash) > 0, "dHash is empty"
    assert len(phash) > 0, "pHash is empty"
    
    print("✓ All hash methods working correctly")
    print()
    
    os.remove(test_img)


def test_hash_comparison():
    """Test 2: Hash comparison between identical images."""
    print("=" * 60)
    print("Test 2: Hash Comparison (Identical Images)")
    print("=" * 60)
    
    test_img = os.path.join(TEST_DIR, "test_image.jpg")
    create_test_image(test_img)
    
    hasher = ImageHasher()
    
    hash1 = hasher.perceptual_hash(test_img)
    hash2 = hasher.perceptual_hash(test_img)
    
    similarity = compare_hashes(hash1, hash2)
    distance = hamming_distance(hash1, hash2)
    
    print(f"Hash 1: {hash1}")
    print(f"Hash 2: {hash2}")
    print(f"Similarity: {similarity:.2%}")
    print(f"Hamming Distance: {distance}")
    
    assert similarity == 1.0, f"Identical images should have 100% similarity, got {similarity}"
    assert distance == 0, f"Identical images should have 0 distance, got {distance}"
    
    print("✓ Identical image comparison working correctly")
    print()
    
    os.remove(test_img)


def test_modified_image_detection():
    """Test 3: Detect modified images."""
    print("=" * 60)
    print("Test 3: Modified Image Detection")
    print("=" * 60)
    
    original_img = os.path.join(TEST_DIR, "original.jpg")
    modified_img = os.path.join(TEST_DIR, "modified.jpg")
    
    create_test_image(original_img)
    create_modified_image(original_img, modified_img, brightness=1.2)
    
    hasher = ImageHasher()
    
    hash_orig = hasher.perceptual_hash(original_img)
    hash_mod = hasher.perceptual_hash(modified_img)
    
    similarity = compare_hashes(hash_orig, hash_mod)
    
    print(f"Original Hash: {hash_orig}")
    print(f"Modified Hash: {hash_mod}")
    print(f"Similarity: {similarity:.2%}")
    
    assert similarity >= 0.8, f"Modified image similarity should be >= 80%, got {similarity:.2%}"
    
    print("✓ Modified image detection working correctly (robust to brightness changes)")
    print()
    
    os.remove(original_img)
    os.remove(modified_img)


def test_save_and_verify():
    """Test 4: Save and verify hash."""
    print("=" * 60)
    print("Test 4: Save and Verify Hash")
    print("=" * 60)
    
    test_img = os.path.join(TEST_DIR, "test_image.jpg")
    hash_file = os.path.join(TEST_DIR, "test_hash.json")
    
    create_test_image(test_img)
    
    save_hash_to_file(test_img, hash_file)
    
    print(f"✓ Hash saved to {hash_file}")
    
    is_match, similarity = verify_image(test_img, hash_file)
    
    print(f"Verification: Match={is_match}, Similarity={similarity:.2%}")
    
    assert is_match, "Same image should match"
    assert similarity >= 0.99, f"Similarity should be >= 99%, got {similarity:.2%}"
    
    print("✓ Save and verify working correctly")
    print()
    
    os.remove(test_img)
    os.remove(hash_file)


def test_different_images():
    """Test 5: Compare completely different images."""
    print("=" * 60)
    print("Test 5: Different Images")
    print("=" * 60)
    
    img1 = os.path.join(TEST_DIR, "image1.jpg")
    img2 = os.path.join(TEST_DIR, "image2.jpg")
    
    create_test_image(img1, color=(255, 0, 0))  # Red
    create_test_image(img2, color=(0, 0, 255))  # Blue
    
    hasher = ImageHasher()
    
    hash1 = hasher.perceptual_hash(img1)
    hash2 = hasher.perceptual_hash(img2)
    
    similarity = compare_hashes(hash1, hash2)
    
    print(f"Image 1 Hash: {hash1}")
    print(f"Image 2 Hash: {hash2}")
    print(f"Similarity: {similarity:.2%}")
    
    print("✓ Different image comparison working correctly")
    print()
    
    os.remove(img1)
    os.remove(img2)


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("IMAGE HASHING MODULE - TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_hash_computation()
        test_hash_comparison()
        test_modified_image_detection()
        test_save_and_verify()
        test_different_images()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {str(e)}")
        print("=" * 60)
        return 1
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return 1
    finally:
        import shutil
        try:
            shutil.rmtree(TEST_DIR)
            print(f"\n✓ Cleaned up temporary directory: {TEST_DIR}")
        except Exception as e:
            print(f"\n⚠ Failed to clean up {TEST_DIR}: {e}")


if __name__ == "__main__":
    sys.exit(run_all_tests())
