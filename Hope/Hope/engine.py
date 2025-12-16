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
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def cmd_protect(args):
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


def main():
    ensure_utf8_stdout()
    
    parser = argparse.ArgumentParser(
        description="Hope-AD: AI Image Protection System",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
    
    # Fallback arguments for legacy/direct calls
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--target-style', choices=['abstract', 'impressionist', 'cubist', 'sketch', 'watercolor'])
    parser.add_argument('--target', default='')
    parser.add_argument('--target-file')
    parser.add_argument('--nightshade', action='store_true')
    parser.add_argument('--source-concept', default='artwork')
    parser.add_argument('--target-concept', default='noise')
    parser.add_argument('--intensity', type=float, default=0.30)
    parser.add_argument('--iterations', type=int, default=150)
    parser.add_argument('--output-quality', type=int, default=92)
    parser.add_argument('--max-file-size', type=int)
    
    args = parser.parse_args()
    
    if args.command == 'protect':
        return cmd_protect(args)
    
    if args.input and args.output:
        return cmd_protect(args)
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
