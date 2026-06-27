# Dataset Instructions

## Dataset: 2018 Data Science Bowl - Nuclei Segmentation

**Source:** [Kaggle Competition](https://www.kaggle.com/c/data-science-bowl-2018)

**Description:** This dataset contains microscopy images of cell nuclei from diverse conditions (different cell types, magnifications, and imaging modalities). The goal is to segment individual nuclei in these images.

### Dataset Statistics
- **Training images:** ~670 images (we'll use a subset of 200-500 for this project)
- **Image sizes:** Variable (mostly 256x256 to 1024x1024)
- **Format:** PNG images with corresponding binary masks
- **Challenge:** Diverse imaging conditions make this a robust test for segmentation models

## How to Download

### Option 1: Kaggle API (Recommended)

1. Install the Kaggle package:
   ```bash
   pip install kaggle
   ```

2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token" to download `kaggle.json`
   - Place it in:
     - **Linux/Mac:** `~/.kaggle/kaggle.json`
     - **Windows:** `C:\Users\<username>\.kaggle\kaggle.json`
   - On Linux/Mac, set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. Run the download script:
   ```bash
   python download_data.py
   ```

### Option 2: Manual Download

1. Visit https://www.kaggle.com/c/data-science-bowl-2018/data
2. Accept competition rules (you may need to create a Kaggle account)
3. Download `data-science-bowl-2018.zip`
4. Extract it into this `data/` directory

### Expected Directory Structure

After downloading and extraction:

```
data/
├── stage1_train/
│   ├── 0a7c6bfb3f5c6b2e1a3c5e7d8f9a1b2c/
│   │   ├── images/
│   │   │   └── 0a7c6bfb3f5c6b2e1a3c5e7d8f9a1b2c.png
│   │   └── masks/
│   │       ├── 123abc.png
│   │       ├── 456def.png
│   │       └── ...
│   └── [more image folders...]
├── stage1_test/
└── stage2_test_final/
```

**Note:** We use `stage1_train/` which contains both images and ground truth masks.

## Dataset Preprocessing

The `dataset.py` script will:
1. Load images and combine multiple mask files into a single binary mask
2. Resize all images to 256x256 for consistent training
3. Normalize pixel values to [0, 1]
4. Apply data augmentation (random flips, rotations)
5. Split into train/validation sets (80/20)
