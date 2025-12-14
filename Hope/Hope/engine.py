#!/usr/bin/env python3
"""
Hope-AD Engine - Command Line Interface
Unified CLI for all image protection and detection features.
"""

import sys
import argparse
from pathlib import Path

try:
    from utils import ensure_utf8_stdout, println
except ImportError:
    def println(s):
        print(s)
    def ensure_utf8_stdout():
        pass


def read_target_from_file(path):
    """Read target description from file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def cmd_protect(args):
    """Handle protection commands (glaze, adversarial, nightshade)."""
    if not Path(args.input).exists():
        println(f"ERROR: File not found: {args.input}")
        return 1
    
    try:
        if args.target_style:
            println(f"STATUS: Using Glaze-style protection (target: {args.target_style})")
            from glaze_protection import GlazeStyleProtector
            
            protector = GlazeStyleProtector(
                target_style=args.target_style,
                intensity=args.intensity,
                iterations=args.iterations
            )
            success = protector.protect_image(
                args.input, 
                args.output, 
                output_quality=args.output_quality
            )
            
        elif args.nightshade:
            println(f"STATUS: Using Nightshade protection")
            from nightshade_protection import NightshadeProtector
            
            protector = NightshadeProtector(
                source_concept=args.source_concept or "artwork",
                target_concept=args.target_concept or "noise",
                intensity=args.intensity,
                iterations=args.iterations
            )
            success = protector.protect_image(
                args.input,
                args.output,
                output_quality=args.output_quality
            )
            
        else:
            println("STATUS: Using adversarial perturbations")
            from adversarial_perturbations import AdversarialProtector
            
            target_description = args.target or ""
            if args.target_file:
                target_description = read_target_from_file(args.target_file)
            
            protector = AdversarialProtector(
                intensity=args.intensity,
                iterations=args.iterations
            )
            success = protector.protect_image(
                args.input, 
                args.output, 
                target_description,
                output_quality=args.output_quality,
                max_file_size_kb=args.max_file_size
            )
        
        return 0 if success else 1
        
    except Exception as e:
        println(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_detect(args):
    """Handle AI detection command."""
    if not Path(args.input).exists():
        println(f"ERROR: File not found: {args.input}")
        return 1
    
    try:
        from ai_detector import AIDetector, DetectionMethod
        
        method = DetectionMethod.COMBINED
        if args.method == 'statistical':
            method = DetectionMethod.STATISTICAL
        elif args.method == 'cnn':
            method = DetectionMethod.CNN
        
        detector = AIDetector(verbose=True)
        result = detector.detect(args.input, method=method, threshold=args.threshold)
        
        println("")
        println(f"=== DETECTION RESULT ===")
        println(f"AI Generated: {'YES' if result.is_ai_generated else 'NO'}")
        println(f"Confidence: {result.confidence:.1%}")
        println(f"Method: {result.method}")
        
        return 0
        
    except Exception as e:
        println(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_embed(args):
    """Handle steganography/watermark embedding."""
    if not Path(args.input).exists():
        println(f"ERROR: File not found: {args.input}")
        return 1
    
    try:
        if args.watermark:
            from robust_watermark import RobustWatermarker
            
            watermarker = RobustWatermarker(verbose=True)
            success = watermarker.embed_to_file(
                args.input, 
                args.output, 
                key=args.key,
                quality=args.output_quality
            )
        else:
            from steganography import CopyrightEmbedder
            
            method = 'dct' if args.dct else 'lsb'
            embedder = CopyrightEmbedder(method=method, verbose=True)
            success = embedder.embed_to_file(
                args.input,
                args.output,
                owner=args.owner or "Unknown",
                quality=args.output_quality
            )
        
        return 0 if success else 1
        
    except Exception as e:
        println(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_extract(args):
    """Handle extraction of embedded data."""
    if not Path(args.input).exists():
        println(f"ERROR: File not found: {args.input}")
        return 1
    
    try:
        if args.watermark:
            from robust_watermark import RobustWatermarker
            
            watermarker = RobustWatermarker(verbose=True)
            detected, details = watermarker.detect(args.input, key=args.key)
            
            println(f"\nWatermark detected: {'YES' if detected else 'NO'}")
        else:
            from steganography import CopyrightEmbedder
            
            method = 'dct' if args.dct else 'lsb'
            embedder = CopyrightEmbedder(method=method, verbose=True)
            data = embedder.extract_copyright(args.input)
            
            if data:
                println(f"\nExtracted data: {data}")
            else:
                println("\nNo embedded data found")
        
        return 0
        
    except Exception as e:
        println(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    ensure_utf8_stdout()
    
    parser = argparse.ArgumentParser(
        description="Hope-AD: AI Image Protection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Glaze protection
  python engine.py protect -i input.jpg -o output.jpg --target-style abstract
  
  # Nightshade protection
  python engine.py protect -i input.jpg -o output.jpg --nightshade --source-concept dog --target-concept cat
  
  # AI detection
  python engine.py detect -i image.jpg
  
  # Embed copyright
  python engine.py embed -i input.jpg -o output.jpg --owner "Artist Name"
  
  # Extract copyright
  python engine.py extract -i image.jpg
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    protect_parser = subparsers.add_parser('protect', help='Protect image from AI')
    protect_parser.add_argument('-i', '--input', required=True, help='Input image')
    protect_parser.add_argument('-o', '--output', required=True, help='Output image')
    protect_parser.add_argument('--target-style', choices=['abstract', 'impressionist', 'cubist', 'sketch', 'watercolor'],
                               help='Glaze-style protection')
    protect_parser.add_argument('--nightshade', action='store_true', help='Use Nightshade protection')
    protect_parser.add_argument('--source-concept', default='artwork', help='Source concept (for Nightshade)')
    protect_parser.add_argument('--target-concept', default='noise', help='Target concept (for Nightshade)')
    protect_parser.add_argument('--target', default='', help='Target description (for adversarial)')
    protect_parser.add_argument('--target-file', help='Read target from file')
    protect_parser.add_argument('--intensity', type=float, default=0.30, help='Protection intensity')
    protect_parser.add_argument('--iterations', type=int, default=150, help='Optimization iterations')
    protect_parser.add_argument('--output-quality', type=int, default=92, help='JPEG quality')
    protect_parser.add_argument('--max-file-size', type=int, help='Max file size in KB')
    
    detect_parser = subparsers.add_parser('detect', help='Detect if image is AI-generated')
    detect_parser.add_argument('-i', '--input', required=True, help='Input image')
    detect_parser.add_argument('--method', choices=['statistical', 'cnn', 'combined'], default='combined',
                              help='Detection method')
    detect_parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    
    embed_parser = subparsers.add_parser('embed', help='Embed copyright/watermark')
    embed_parser.add_argument('-i', '--input', required=True, help='Input image')
    embed_parser.add_argument('-o', '--output', required=True, help='Output image')
    embed_parser.add_argument('--owner', help='Copyright owner')
    embed_parser.add_argument('--key', default='hope-ad', help='Watermark key')
    embed_parser.add_argument('--watermark', action='store_true', help='Use robust watermark')
    embed_parser.add_argument('--dct', action='store_true', help='Use DCT steganography')
    embed_parser.add_argument('--output-quality', type=int, default=95, help='JPEG quality')
    
    extract_parser = subparsers.add_parser('extract', help='Extract embedded data')
    extract_parser.add_argument('-i', '--input', required=True, help='Input image')
    extract_parser.add_argument('--key', default='hope-ad', help='Watermark key')
    extract_parser.add_argument('--watermark', action='store_true', help='Detect robust watermark')
    extract_parser.add_argument('--dct', action='store_true', help='Extract DCT steganography')
    
    parser.add_argument('--input', help='Input image (legacy)')
    parser.add_argument('--output', help='Output image (legacy)')
    parser.add_argument('--target-style', choices=['abstract', 'impressionist', 'cubist', 'sketch', 'watercolor'])
    parser.add_argument('--target', default='')
    parser.add_argument('--target-file')
    parser.add_argument('--intensity', type=float, default=0.30)
    parser.add_argument('--iterations', type=int, default=150)
    parser.add_argument('--output-quality', type=int, default=92)
    parser.add_argument('--max-file-size', type=int)
    
    args = parser.parse_args()
    
    if args.command == 'protect':
        return cmd_protect(args)
    elif args.command == 'detect':
        return cmd_detect(args)
    elif args.command == 'embed':
        return cmd_embed(args)
    elif args.command == 'extract':
        return cmd_extract(args)
    
    if args.input and args.output:
        return cmd_protect(args)
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
