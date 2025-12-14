#!/usr/bin/env python3
"""
Steganography Module for Hope-AD
Embeds copyright and ownership data invisibly into images.
Supports LSB (Least Significant Bit) and DCT (Discrete Cosine Transform) methods.
"""

import sys
import io
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict, Union
import json
import hashlib
from datetime import datetime

try:
    from scipy.fftpack import dct, idct
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from utils import println, ensure_utf8_stdout
except ImportError:
    def println(s):
        sys.stdout.write(str(s) + "\n")
        sys.stdout.flush()
    def ensure_utf8_stdout():
        pass


class LSBSteganography:
    """
    Least Significant Bit steganography.
    Embeds data in the least significant bits of pixel values.
    """
    
    MAGIC = b'\x89HOP'
    
    def __init__(self, bits_per_channel: int = 2, verbose: bool = True):
        """
        Initialize LSB steganography.
        
        Args:
            bits_per_channel: Number of LSB bits to use (1-4)
            verbose: Print status messages
        """
        self.bits_per_channel = min(max(bits_per_channel, 1), 4)
        self.verbose = verbose
    
    def _text_to_bits(self, text: str) -> str:
        """Convert text to binary string."""
        text_bytes = text.encode('utf-8')
        return ''.join(format(byte, '08b') for byte in text_bytes)
    
    def _bits_to_text(self, bits: str) -> str:
        """Convert binary string to text."""
        bytes_list = [int(bits[i:i+8], 2) for i in range(0, len(bits), 8)]
        return bytes(bytes_list).decode('utf-8', errors='ignore')
    
    def _data_to_bits(self, data: bytes) -> str:
        """Convert bytes to binary string."""
        return ''.join(format(byte, '08b') for byte in data)
    
    def _bits_to_data(self, bits: str) -> bytes:
        """Convert binary string to bytes."""
        bytes_list = [int(bits[i:i+8], 2) for i in range(0, len(bits), 8)]
        return bytes(bytes_list)
    
    def get_capacity(self, image: Image.Image) -> int:
        """
        Get the data capacity of an image in bytes.
        
        Args:
            image: PIL Image
            
        Returns:
            Capacity in bytes
        """
        width, height = image.size
        channels = len(image.getbands())
        
        total_bits = width * height * channels * self.bits_per_channel
        
        header_bits = (len(self.MAGIC) + 4) * 8  # magic + 4 bytes for length
        
        available_bits = total_bits - header_bits
        return available_bits // 8
    
    def embed(self, image: Image.Image, data: Union[str, bytes, dict]) -> Image.Image:
        """
        Embed data into image using LSB method.
        
        Args:
            image: Source PIL Image
            data: Data to embed (string, bytes, or dict)
            
        Returns:
            Image with embedded data
        """
        if isinstance(data, dict):
            data_bytes = json.dumps(data).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        capacity = self.get_capacity(image)
        if len(data_bytes) > capacity:
            raise ValueError(f"Data too large: {len(data_bytes)} bytes, capacity: {capacity} bytes")
        
        header = self.MAGIC + len(data_bytes).to_bytes(4, 'big')
        full_data = header + data_bytes
        
        bits = self._data_to_bits(full_data)
        
        img_array = np.array(image.convert('RGB'))
        flat = img_array.flatten()
        
        mask = 0xFF << self.bits_per_channel
        
        bit_idx = 0
        for i in range(len(flat)):
            if bit_idx >= len(bits):
                break
            
            flat[i] = flat[i] & mask
            
            end_idx = min(bit_idx + self.bits_per_channel, len(bits))
            bit_chunk = bits[bit_idx:end_idx]
            
            while len(bit_chunk) < self.bits_per_channel:
                bit_chunk += '0'
            
            flat[i] = flat[i] | int(bit_chunk, 2)
            bit_idx += self.bits_per_channel
        
        img_array = flat.reshape(img_array.shape)
        result = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
        
        if self.verbose:
            println(f"Embedded {len(data_bytes)} bytes using LSB-{self.bits_per_channel}")
        
        return result
    
    def extract(self, image: Image.Image) -> Optional[bytes]:
        """
        Extract embedded data from image.
        
        Args:
            image: Image with embedded data
            
        Returns:
            Extracted bytes or None if no data found
        """
        img_array = np.array(image.convert('RGB'))
        flat = img_array.flatten()
        
        bits = ''
        lsb_mask = (1 << self.bits_per_channel) - 1
        
        for pixel in flat:
            extracted = pixel & lsb_mask
            bits += format(extracted, f'0{self.bits_per_channel}b')
        
        magic_bits = len(self.MAGIC) * 8
        header_bytes = self._bits_to_data(bits[:magic_bits])
        
        if header_bytes != self.MAGIC:
            if self.verbose:
                println("No embedded data found (magic mismatch)")
            return None
        
        length_start = magic_bits
        length_end = length_start + 32  # 4 bytes
        length_bits = bits[length_start:length_end]
        data_length = int(length_bits, 2)
        
        if data_length <= 0 or data_length > len(flat):
            if self.verbose:
                println("Invalid data length")
            return None
        
        data_start = length_end
        data_end = data_start + (data_length * 8)
        data_bits = bits[data_start:data_end]
        
        extracted_data = self._bits_to_data(data_bits)
        
        if self.verbose:
            println(f"Extracted {len(extracted_data)} bytes")
        
        return extracted_data


class DCTSteganography:
    """
    DCT domain steganography.
    Embeds data in DCT coefficients for JPEG robustness.
    """
    
    MAGIC = b'\x89DCT'
    
    def __init__(self, block_size: int = 8, strength: float = 0.1, verbose: bool = True):
        """
        Initialize DCT steganography.
        
        Args:
            block_size: DCT block size (usually 8)
            strength: Embedding strength
            verbose: Print status
        """
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is required for DCT steganography")
        
        self.block_size = block_size
        self.strength = strength
        self.verbose = verbose
    
    def _dct2(self, block: np.ndarray) -> np.ndarray:
        """2D DCT."""
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def _idct2(self, block: np.ndarray) -> np.ndarray:
        """2D inverse DCT."""
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    
    def embed(self, image: Image.Image, data: Union[str, bytes, dict]) -> Image.Image:
        """
        Embed data using DCT method.
        
        Args:
            image: Source image
            data: Data to embed
            
        Returns:
            Image with embedded data
        """
        if isinstance(data, dict):
            data_bytes = json.dumps(data).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        header = self.MAGIC + len(data_bytes).to_bytes(4, 'big')
        full_data = header + data_bytes
        
        bits = ''.join(format(byte, '08b') for byte in full_data)
        
        img_array = np.array(image.convert('RGB'), dtype=np.float64)
        
        y = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        
        h, w = y.shape
        num_blocks_h = h // self.block_size
        num_blocks_w = w // self.block_size
        
        bit_idx = 0
        
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                if bit_idx >= len(bits):
                    break
                
                block = y[i*self.block_size:(i+1)*self.block_size,
                         j*self.block_size:(j+1)*self.block_size]
                
                dct_block = self._dct2(block)
                
                bit = int(bits[bit_idx])
                
                coef_val = dct_block[3, 4]
                if bit == 1:
                    dct_block[3, 4] = abs(coef_val) * (1 + self.strength)
                else:
                    dct_block[3, 4] = abs(coef_val) * (1 - self.strength)
                
                y[i*self.block_size:(i+1)*self.block_size,
                  j*self.block_size:(j+1)*self.block_size] = self._idct2(dct_block)
                
                bit_idx += 1
        
        diff = y - (0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2])
        img_array[:,:,0] = np.clip(img_array[:,:,0] + diff, 0, 255)
        img_array[:,:,1] = np.clip(img_array[:,:,1] + diff, 0, 255)
        img_array[:,:,2] = np.clip(img_array[:,:,2] + diff, 0, 255)
        
        result = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
        
        if self.verbose:
            println(f"DCT embedded {len(data_bytes)} bytes")
        
        return result
    
    def extract(self, image: Image.Image) -> Optional[bytes]:
        """Extract DCT-embedded data."""
        img_array = np.array(image.convert('RGB'), dtype=np.float64)
        
        y = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        
        h, w = y.shape
        num_blocks_h = h // self.block_size
        num_blocks_w = w // self.block_size
        
        bits = ''
        
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                block = y[i*self.block_size:(i+1)*self.block_size,
                         j*self.block_size:(j+1)*self.block_size]
                
                dct_block = self._dct2(block)
                
                coef_val = dct_block[3, 4]
                bits += '1' if coef_val > 0 else '0'
        
        try:
            magic_bits = len(self.MAGIC) * 8
            header_bytes = bytes(int(bits[i:i+8], 2) for i in range(0, magic_bits, 8))
            
            if header_bytes != self.MAGIC:
                if self.verbose:
                    println("No DCT embedded data found")
                return None
            
            length_bits = bits[magic_bits:magic_bits+32]
            data_length = int(length_bits, 2)
            
            if data_length <= 0:
                return None
            
            data_start = magic_bits + 32
            data_end = data_start + (data_length * 8)
            data_bits = bits[data_start:data_end]
            
            extracted = bytes(int(data_bits[i:i+8], 2) for i in range(0, len(data_bits), 8))
            
            if self.verbose:
                println(f"DCT extracted {len(extracted)} bytes")
            
            return extracted
            
        except Exception as e:
            if self.verbose:
                println(f"DCT extraction error: {e}")
            return None


class CopyrightEmbedder:
    """
    High-level interface for embedding copyright information.
    """
    
    def __init__(self, method: str = 'lsb', verbose: bool = True):
        """
        Initialize copyright embedder.
        
        Args:
            method: 'lsb' or 'dct'
            verbose: Print status
        """
        self.method = method.lower()
        self.verbose = verbose
        
        if self.method == 'lsb':
            self.steganographer = LSBSteganography(bits_per_channel=2, verbose=verbose)
        elif self.method == 'dct':
            self.steganographer = DCTSteganography(verbose=verbose)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'lsb' or 'dct'")
    
    def embed_copyright(
        self,
        image: Union[str, Image.Image],
        owner: str,
        license_type: str = "All Rights Reserved",
        custom_data: Optional[dict] = None
    ) -> Image.Image:
        """
        Embed copyright information into image.
        
        Args:
            image: Image or path
            owner: Copyright owner name
            license_type: License type
            custom_data: Additional metadata
            
        Returns:
            Image with embedded copyright
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        copyright_data = {
            "owner": owner,
            "license": license_type,
            "timestamp": datetime.now().isoformat(),
            "checksum": hashlib.sha256(owner.encode()).hexdigest()[:16]
        }
        
        if custom_data:
            copyright_data.update(custom_data)
        
        if self.verbose:
            println(f"Embedding copyright: {owner}")
        
        return self.steganographer.embed(image, copyright_data)
    
    def extract_copyright(self, image: Union[str, Image.Image]) -> Optional[dict]:
        """
        Extract copyright information from image.
        
        Args:
            image: Image or path
            
        Returns:
            Copyright data dict or None
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        data = self.steganographer.extract(image)
        
        if data is None:
            return None
        
        try:
            return json.loads(data.decode('utf-8'))
        except:
            return {"raw": data.decode('utf-8', errors='ignore')}
    
    def embed_to_file(
        self,
        input_path: str,
        output_path: str,
        owner: str,
        license_type: str = "All Rights Reserved",
        quality: int = 95
    ) -> bool:
        """
        Embed copyright and save to file.
        
        Args:
            input_path: Source image path
            output_path: Output image path
            owner: Copyright owner
            license_type: License type
            quality: JPEG quality
            
        Returns:
            True if successful
        """
        try:
            image = Image.open(input_path).convert('RGB')
            result = self.embed_copyright(image, owner, license_type)
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.lower().endswith('.png'):
                result.save(output_path, format='PNG')
            else:
                result.save(output_path, format='JPEG', quality=quality)
            
            if self.verbose:
                println(f"Saved: {output_path}")
            
            return True
            
        except Exception as e:
            println(f"Error: {e}")
            return False


def embed_copyright(
    input_path: str,
    output_path: str,
    owner: str,
    method: str = 'lsb',
    verbose: bool = True
) -> bool:
    """
    Convenience function to embed copyright.
    
    Args:
        input_path: Source image
        output_path: Output image
        owner: Copyright owner
        method: 'lsb' or 'dct'
        verbose: Print status
        
    Returns:
        True if successful
    """
    embedder = CopyrightEmbedder(method=method, verbose=verbose)
    return embedder.embed_to_file(input_path, output_path, owner)


def extract_copyright(image_path: str, method: str = 'lsb') -> Optional[dict]:
    """
    Convenience function to extract copyright.
    
    Args:
        image_path: Image to check
        method: 'lsb' or 'dct'
        
    Returns:
        Copyright data or None
    """
    embedder = CopyrightEmbedder(method=method, verbose=False)
    return embedder.extract_copyright(image_path)


if __name__ == "__main__":
    ensure_utf8_stdout()
    
    print("Steganography Module")
    print("=" * 40)
    print("\nMethods: LSB (fragile), DCT (JPEG-robust)")
    print("\nUsage:")
    print("  from steganography import embed_copyright, extract_copyright")
    print("  embed_copyright('input.jpg', 'output.jpg', 'Artist Name', method='lsb')")
    print("  data = extract_copyright('output.jpg', method='lsb')")
