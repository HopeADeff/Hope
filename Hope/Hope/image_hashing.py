#!/usr/bin/env python3
import sys
from PIL import Image
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import json


class ImageHasher:
    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size
    
    def _prepare_image(self, image_path: str, size: Tuple[int, int]) -> Image.Image:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize(size, Image.Resampling.LANCZOS)
        return img
    
    def average_hash(self, image_path: str) -> str:
        img = self._prepare_image(image_path, (self.hash_size, self.hash_size))
        pixels = np.array(img, dtype=np.float32)
        
        avg = pixels.mean()
        hash_bits = (pixels > avg).flatten()
        
        return self._bits_to_hex(hash_bits)
    
    def difference_hash(self, image_path: str) -> str:
        img = self._prepare_image(image_path, (self.hash_size + 1, self.hash_size))
        pixels = np.array(img, dtype=np.float32)
        
        diff = pixels[:, 1:] > pixels[:, :-1]
        hash_bits = diff.flatten()
        
        return self._bits_to_hex(hash_bits)
    
    def perceptual_hash(self, image_path: str) -> str:
        img = self._prepare_image(image_path, (32, 32))
        pixels = np.array(img, dtype=np.float32)
        
        dct = self._dct2d(pixels)
        
        dct_low = dct[:self.hash_size, :self.hash_size]
        
        median = np.median(dct_low)
        hash_bits = (dct_low > median).flatten()
        
        return self._bits_to_hex(hash_bits)
    
    def _dct2d(self, img: np.ndarray) -> np.ndarray:
        try:
            from scipy.fftpack import dct
            return dct(dct(img.T, norm='ortho').T, norm='ortho')
        except ImportError:
            return self._simple_dct2d(img)
    
    def _simple_dct2d(self, img: np.ndarray) -> np.ndarray:
        M, N = img.shape
        dct_matrix = np.zeros((M, N))
        
        for u in range(M):
            for v in range(N):
                sum_val = 0
                for i in range(M):
                    for j in range(N):
                        sum_val += img[i, j] * \
                                   np.cos((2 * i + 1) * u * np.pi / (2 * M)) * \
                                   np.cos((2 * j + 1) * v * np.pi / (2 * N))
                
                cu = 1 / np.sqrt(M) if u == 0 else np.sqrt(2 / M)
                cv = 1 / np.sqrt(N) if v == 0 else np.sqrt(2 / N)
                dct_matrix[u, v] = cu * cv * sum_val
        
        return dct_matrix
    
    def _bits_to_hex(self, bits: np.ndarray) -> str:
        binary_str = ''.join(['1' if b else '0' for b in bits])
        
        while len(binary_str) % 4 != 0:
            binary_str += '0'
        
        hex_str = hex(int(binary_str, 2))[2:]
        
        expected_len = (len(bits) + 3) // 4
        return hex_str.zfill(expected_len)
    
    def compute_hash(self, image_path: str, method: str = "phash") -> str:
        method = method.lower()
        
        if method == "ahash":
            return self.average_hash(image_path)
        elif method == "dhash":
            return self.difference_hash(image_path)
        elif method == "phash":
            return self.perceptual_hash(image_path)
        else:
            raise ValueError(f"Unknown hash method: {method}. Use 'ahash', 'dhash', or 'phash'")
    
    def compute_all_hashes(self, image_path: str) -> dict:
        return {
            "ahash": self.average_hash(image_path),
            "dhash": self.difference_hash(image_path),
            "phash": self.perceptual_hash(image_path)
        }


def hamming_distance(hash1: str, hash2: str) -> int:
    if len(hash1) != len(hash2):
        raise ValueError("Hashes must be the same length")
    
    int1 = int(hash1, 16)
    int2 = int(hash2, 16)
    xor = int1 ^ int2
    
    distance = bin(xor).count('1')
    return distance


def compare_hashes(hash1: str, hash2: str, max_bits: int = 64) -> float:
    distance = hamming_distance(hash1, hash2)
    similarity = 1.0 - (distance / max_bits)
    return similarity


def is_similar(hash1: str, hash2: str, threshold: float = 0.9, max_bits: int = 64) -> bool:
    similarity = compare_hashes(hash1, hash2, max_bits)
    return similarity >= threshold


def save_hash_to_file(image_path: str, output_path: str, method: str = "phash"):
    hasher = ImageHasher()
    hash_value = hasher.compute_hash(image_path, method)
    
    hash_data = {
        "image_path": str(Path(image_path).absolute()),
        "method": method,
        "hash": hash_value,
        "hash_size": hasher.hash_size
    }
    
    with open(output_path, 'w') as f:
        json.dump(hash_data, f, indent=2)


def load_hash_from_file(hash_file: str) -> dict:
    with open(hash_file, 'r') as f:
        return json.load(f)


def verify_image(image_path: str, hash_file: str, threshold: float = 0.9) -> Tuple[bool, float]:
    hash_data = load_hash_from_file(hash_file)
    stored_hash = hash_data["hash"]
    method = hash_data["method"]
    hash_size = hash_data.get("hash_size", 8)
    
    hasher = ImageHasher(hash_size=hash_size)
    current_hash = hasher.compute_hash(image_path, method)
    
    max_bits = hash_size * hash_size
    similarity = compare_hashes(stored_hash, current_hash, max_bits)
    is_match = similarity >= threshold
    
    return is_match, similarity


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Image Hashing Tool")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", help="Output hash file (JSON)")
    parser.add_argument("--method", default="phash", choices=["ahash", "dhash", "phash"],
                        help="Hash method to use")
    parser.add_argument("--compare", help="Compare with another image")
    parser.add_argument("--verify", help="Verify against a hash file")
    parser.add_argument("--threshold", type=float, default=0.9, 
                        help="Similarity threshold for verification")
    
    args = parser.parse_args()
    
    hasher = ImageHasher()
    
    if args.compare:
        hash1 = hasher.compute_hash(args.input, args.method)
        hash2 = hasher.compute_hash(args.compare, args.method)
        similarity = compare_hashes(hash1, hash2)
        
        print(f"Image 1: {args.input}")
        print(f"Hash 1:  {hash1}")
        print(f"Image 2: {args.compare}")
        print(f"Hash 2:  {hash2}")
        print(f"Similarity: {similarity:.2%}")
        print(f"Hamming Distance: {hamming_distance(hash1, hash2)}")
        
    elif args.verify:
        is_match, similarity = verify_image(args.input, args.verify, args.threshold)
        
        print(f"Image: {args.input}")
        print(f"Hash file: {args.verify}")
        print(f"Similarity: {similarity:.2%}")
        print(f"Match: {'YES' if is_match else 'NO'} (threshold: {args.threshold:.2%})")
        
        sys.exit(0 if is_match else 1)
        
    else:
        if args.output:
            save_hash_to_file(args.input, args.output, args.method)
            print(f"Hash saved to: {args.output}")
        
        hash_value = hasher.compute_hash(args.input, args.method)
        print(f"Image: {args.input}")
        print(f"Method: {args.method}")
        print(f"Hash: {hash_value}")


if __name__ == "__main__":
    main()
