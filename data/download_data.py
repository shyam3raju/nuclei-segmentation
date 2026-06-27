"""
Dataset download script for Kaggle 2018 Data Science Bowl nuclei dataset.

This script provides instructions and code to download the nuclei segmentation dataset.
We use a subset of ~200-500 images for this portfolio project to keep training time reasonable.

Dataset: 2018 Data Science Bowl - Find the nuclei in divergent images
Source: https://www.kaggle.com/c/data-science-bowl-2018
"""

import os
import zipfile
from pathlib import Path

def download_with_kaggle_api():
    """
    Download dataset using Kaggle API.
    
    Prerequisites:
    1. Install kaggle package: pip install kaggle
    2. Set up Kaggle API credentials:
       - Go to https://www.kaggle.com/account
       - Click "Create New API Token" to download kaggle.json
       - Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)
       - On Linux/Mac: chmod 600 ~/.kaggle/kaggle.json
    
    Usage:
        python download_data.py
    """
    print("Downloading 2018 Data Science Bowl dataset from Kaggle...")
    
    # Create data directory
    data_dir = Path(__file__).parent
    data_dir.mkdir(exist_ok=True)
    
    # Download using Kaggle API
    import kaggle
    
    # Download competition dataset
    kaggle.api.competition_download_files(
        'data-science-bowl-2018',
        path=str(data_dir),
        quiet=False
    )
    
    # Extract the dataset
    zip_path = data_dir / 'data-science-bowl-2018.zip'
    if zip_path.exists():
        print(f"\nExtracting dataset to {data_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete!")
        
        # Clean up zip file
        zip_path.unlink()
        print(f"Removed {zip_path}")
    
    print("\nDataset structure:")
    print("data/")
    print("├── stage1_train/  (training images and masks)")
    print("├── stage1_test/   (test images)")
    print("└── stage2_test_final/  (final test images)")
    print("\nFor this project, we'll use stage1_train which contains ~670 images with masks.")

def manual_download_instructions():
    """
    Print manual download instructions if Kaggle API is not set up.
    """
    print("=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print("\nIf you don't have Kaggle API set up, follow these steps:")
    print("\n1. Go to: https://www.kaggle.com/c/data-science-bowl-2018/data")
    print("2. Accept the competition rules (if prompted)")
    print("3. Download the dataset (data-science-bowl-2018.zip)")
    print("4. Extract the zip file into this 'data/' directory")
    print("\nAfter extraction, your structure should look like:")
    print("data/")
    print("├── stage1_train/")
    print("│   ├── <image_id_1>/")
    print("│   │   ├── images/")
    print("│   │   │   └── <image_id_1>.png")
    print("│   │   └── masks/")
    print("│   │       ├── mask1.png")
    print("│   │       ├── mask2.png")
    print("│   │       └── ...")
    print("│   └── ...")
    print("=" * 70)

if __name__ == "__main__":
    try:
        download_with_kaggle_api()
    except ImportError:
        print("Kaggle package not found. Please install it with: pip install kaggle")
        print()
        manual_download_instructions()
    except Exception as e:
        print(f"Error downloading with Kaggle API: {e}")
        print()
        manual_download_instructions()
