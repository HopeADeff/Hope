#!/usr/bin/env python3
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image, ImageDraw
import numpy as np

try:
    from glaze_protection import GlazeStyleProtector
    GLAZE_AVAILABLE = True
except ImportError:
    GLAZE_AVAILABLE = False


def create_test_image(path, size=(256, 256)):
    img = Image.new('RGB', size, (100, 150, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 150], fill=(200, 100, 100))
    draw.ellipse([100, 100, 200, 200], fill=(100, 200, 100))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def test_style_embeddings_structure():
    print("=" * 60)
    print("Test 1: Style Embeddings Structure")
    print("=" * 60)
    
    if not GLAZE_AVAILABLE:
        print("⚠ Glaze protection not available, skipping test")
        return True
    
    try:
        protector = GlazeStyleProtector(target_style="abstract", verbose=False)
        
        assert isinstance(protector.style_embeddings, dict), "style_embeddings should be a dict"
        
        for style_name in protector.STYLE_DESCRIPTIONS.keys():
            assert style_name in protector.style_embeddings, f"Missing embeddings for {style_name}"
            style_dict = protector.style_embeddings[style_name]
            
            assert isinstance(style_dict, dict), f"Embeddings for {style_name} should be a dict"
            assert 'base' in style_dict, f"Missing 'base' embeddings for {style_name}"
            
            base_embedding = style_dict['base']
            assert isinstance(base_embedding, torch.Tensor), "Base embedding should be a tensor"
            assert base_embedding.shape[-1] == protector.CLIP_BASE_DIM, f"Base embedding should be {protector.CLIP_BASE_DIM}-dim, got {base_embedding.shape[-1]}"
            
            if protector.has_large:
                assert 'large' in style_dict, f"Missing 'large' embeddings for {style_name}"
                large_embedding = style_dict['large']
                assert isinstance(large_embedding, torch.Tensor), "Large embedding should be a tensor"
                assert large_embedding.shape[-1] == protector.CLIP_LARGE_DIM, f"Large embedding should be {protector.CLIP_LARGE_DIM}-dim, got {large_embedding.shape[-1]}"
                print(f"✓ {style_name}: base=512-dim, large=768-dim")
            else:
                print(f"✓ {style_name}: base=512-dim (large model not available)")
        
        print("✓ Style embeddings structure is correct")
        print()
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_style_embedding():
    """Test 2: Verify get_style_embedding works correctly."""
    print("=" * 60)
    print("Test 2: Get Style Embedding")
    print("=" * 60)
    
    if not GLAZE_AVAILABLE:
        print("⚠ Glaze protection not available, skipping test")
        return True
    
    try:
        protector = GlazeStyleProtector(target_style="abstract", verbose=False)
        
        base_embedding = protector.get_style_embedding("abstract", "base")
        assert base_embedding is not None, "Base embedding should not be None"
        assert base_embedding.shape[-1] == protector.CLIP_BASE_DIM, f"Base embedding should be {protector.CLIP_BASE_DIM}-dim, got {base_embedding.shape[-1]}"
        print(f"✓ get_style_embedding('abstract', 'base'): {base_embedding.shape}")
        
        if protector.has_large:
            large_embedding = protector.get_style_embedding("abstract", "large")
            assert large_embedding is not None, "Large embedding should not be None"
            assert large_embedding.shape[-1] == protector.CLIP_LARGE_DIM, f"Large embedding should be {protector.CLIP_LARGE_DIM}-dim, got {large_embedding.shape[-1]}"
            print(f"✓ get_style_embedding('abstract', 'large'): {large_embedding.shape}")
        else:
            print("⚠ Large model not available, skipping large embedding test")
        
        print("✓ get_style_embedding works correctly")
        print()
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_style_loss_dimensions():
    """Test 3: Verify style_loss works with both models."""
    print("=" * 60)
    print("Test 3: Style Loss Dimension Matching")
    print("=" * 60)
    
    if not GLAZE_AVAILABLE:
        print("⚠ Glaze protection not available, skipping test")
        return True
    
    try:
        protector = GlazeStyleProtector(target_style="abstract", verbose=False)
        
        img_tensor = torch.rand(1, 3, 256, 256).to(protector.device)
        
        base_embedding = protector.get_style_embedding("abstract", "base")
        loss_base = protector.style_loss(img_tensor, base_embedding, protector.clip_model)
        assert isinstance(loss_base, torch.Tensor), "Loss should be a tensor"
        print(f"✓ style_loss with ViT-B/32 (512-dim): {loss_base.item():.4f}")
        
        if protector.has_large:
            large_embedding = protector.get_style_embedding("abstract", "large")
            loss_large = protector.style_loss(img_tensor, large_embedding, protector.clip_large)
            assert isinstance(loss_large, torch.Tensor), "Loss should be a tensor"
            print(f"✓ style_loss with ViT-L/14 (768-dim): {loss_large.item():.4f}")
        else:
            print("⚠ Large model not available, skipping large model test")
        
        print("✓ style_loss works correctly with both models")
        print()
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_model_style_attack():
    """Test 4: Verify multi_model_style_attack works without dimension errors."""
    print("=" * 60)
    print("Test 4: Multi-Model Style Attack")
    print("=" * 60)
    
    if not GLAZE_AVAILABLE:
        print("⚠ Glaze protection not available, skipping test")
        return True
    
    try:
        protector = GlazeStyleProtector(target_style="abstract", verbose=False)
        
        img_tensor = torch.rand(1, 3, 256, 256).to(protector.device)
        
        loss = protector.multi_model_style_attack(img_tensor)
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        print(f"✓ multi_model_style_attack completed: loss={loss.item():.4f}")
        
        if protector.has_large:
            print("✓ Both ViT-B/32 and ViT-L/14 models worked correctly")
        else:
            print("✓ ViT-B/32 model worked correctly (ViT-L/14 not available)")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_protection_pipeline():
    """Test 5: Full image protection pipeline (minimal)."""
    print("=" * 60)
    print("Test 5: Full Protection Pipeline")
    print("=" * 60)
    
    if not GLAZE_AVAILABLE:
        print("⚠ Glaze protection not available, skipping test")
        return True
    
    test_dir = tempfile.mkdtemp(prefix="glaze_test_")
    
    try:
        input_path = os.path.join(test_dir, "test_input.jpg")
        output_path = os.path.join(test_dir, "test_output.jpg")
        create_test_image(input_path)
        
        protector = GlazeStyleProtector(
            target_style="abstract", 
            intensity=0.1,
            iterations=5,
            verbose=False
        )
        
        success = protector.protect_image(input_path, output_path)
        
        assert success, "Protection should succeed"
        assert os.path.exists(output_path), "Output file should be created"
        
        output_img = Image.open(output_path)
        assert output_img.size[0] == 256, "Output width should match input"
        assert output_img.size[1] == 256, "Output height should match input"
        
        print(f"✓ Image protection completed successfully")
        print(f"✓ Output saved to: {output_path}")
        print()
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        import shutil
        try:
            shutil.rmtree(test_dir)
        except:
            pass


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GLAZE PROTECTION MODULE - DIMENSION FIX TEST SUITE")
    print("=" * 60 + "\n")
    
    if not GLAZE_AVAILABLE:
        print("=" * 60)
        print("⚠ GLAZE PROTECTION NOT AVAILABLE")
        print("Install CLIP to run tests: pip install git+https://github.com/openai/CLIP.git")
        print("=" * 60)
        return 0
    
    results = []
    results.append(("Style Embeddings Structure", test_style_embeddings_structure()))
    results.append(("Get Style Embedding", test_get_style_embedding()))
    results.append(("Style Loss Dimensions", test_style_loss_dimensions()))
    results.append(("Multi-Model Style Attack", test_multi_model_style_attack()))
    results.append(("Full Protection Pipeline", test_full_protection_pipeline()))
    
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
