"""
Quick setup verification script.

Run this to verify your environment is properly configured before training.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
        
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
        
        import PIL
        print(f"✓ Pillow {PIL.__version__}")
        
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
        
        import albumentations
        print(f"✓ Albumentations {albumentations.__version__}")
        
        import tqdm
        print(f"✓ tqdm")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nPlease install missing packages:")
        print("pip install -r requirements.txt")
        return False


def test_gpu():
    """Check GPU availability."""
    print("\nTesting GPU...")
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        
        # Test GPU memory
        try:
            x = torch.randn(1000, 1000).cuda()
            y = x @ x
            del x, y
            torch.cuda.empty_cache()
            print("  ✓ GPU test successful")
        except Exception as e:
            print(f"  ⚠ GPU test failed: {e}")
    else:
        print("⚠ GPU not available - will use CPU (slower)")
        print("  This is fine for testing, but GPU is recommended for full training")


def test_model():
    """Test model creation."""
    print("\nTesting model...")
    try:
        from model import UNet, count_parameters
        import torch
        
        model = UNet(in_channels=3, out_channels=1)
        n_params = count_parameters(model)
        print(f"✓ U-Net model created successfully")
        print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        
        # Test forward pass
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, 1, 256, 256), f"Wrong output shape: {output.shape}"
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def test_dataset():
    """Test dataset loader."""
    print("\nTesting dataset...")
    
    data_dir = Path("../data/stage1_train")
    
    if not data_dir.exists():
        print(f"⚠ Dataset not found at {data_dir}")
        print("  Please download the dataset first:")
        print("  cd ../data && python download_data.py")
        return False
    
    try:
        from dataset import get_dataloaders
        
        # Test with small subset
        train_loader, val_loader = get_dataloaders(
            data_dir=str(data_dir),
            batch_size=2,
            max_samples=10,
            num_workers=0
        )
        
        # Load one batch
        images, masks = next(iter(train_loader))
        
        print(f"✓ Dataset loader working")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
        print(f"  Batch image shape: {images.shape}")
        print(f"  Batch mask shape: {masks.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("SETUP VERIFICATION")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("GPU", test_gpu),
        ("Model", test_model),
        ("Dataset", test_dataset),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
        print()
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:15} {status}")
    
    all_passed = all(result for _, result in results[:3])  # Exclude dataset test
    
    print("="*60)
    if all_passed:
        print("✓ Setup verification complete!")
        print("\nYou're ready to train:")
        print("  python train.py --num_epochs 30")
    else:
        print("⚠ Some tests failed. Please fix issues before training.")
        print("\nCommon fixes:")
        print("  - Missing packages: pip install -r requirements.txt")
        print("  - Missing dataset: cd ../data && python download_data.py")
    
    if not results[3][1]:  # Dataset test failed
        print("\nNote: Dataset test failed but you can still test the code")
        print("      Download the dataset to enable full training")


if __name__ == "__main__":
    main()
