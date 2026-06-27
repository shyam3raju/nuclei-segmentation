# Setup Guide

Complete setup instructions for the nuclei segmentation project.

## System Requirements

- **Python:** 3.8 or higher
- **RAM:** 8GB minimum (16GB recommended)
- **GPU:** Optional but recommended (NVIDIA GPU with CUDA support)
  - Training on CPU: ~2-3 hours for 30 epochs
  - Training on GPU: ~30-60 minutes for 30 epochs
- **Disk Space:** ~5GB (including dataset)

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/nuclei-segmentation-unet.git
cd nuclei-segmentation-unet
```

### 2. Create Virtual Environment (Recommended)

**Using venv:**
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

**Using conda:**
```bash
conda create -n nuclei python=3.9
conda activate nuclei
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**For GPU support (optional but recommended):**

If you have an NVIDIA GPU, install PyTorch with CUDA:

```bash
# Check your CUDA version first: nvidia-smi

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision
```

Visit [PyTorch website](https://pytorch.org/get-started/locally/) for specific installation commands.

### 4. Download Dataset

**Option A - Using Kaggle API (Recommended):**

1. Install Kaggle:
   ```bash
   pip install kaggle
   ```

2. Get API credentials:
   - Go to https://www.kaggle.com/account
   - Scroll to "API" section
   - Click "Create New API Token"
   - This downloads `kaggle.json`

3. Place credentials:
   - **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`
   - Set permissions (Linux/Mac): `chmod 600 ~/.kaggle/kaggle.json`

4. Download dataset:
   ```bash
   cd data
   python download_data.py
   cd ..
   ```

**Option B - Manual Download:**

1. Visit https://www.kaggle.com/c/data-science-bowl-2018/data
2. Accept competition rules (requires Kaggle account)
3. Download `data-science-bowl-2018.zip`
4. Extract to `data/` directory

### 5. Verify Setup

Test that everything is installed correctly:

```bash
# Test dataset loader
cd src
python dataset.py

# Test model
python model.py
```

You should see output confirming dataset loading and model creation.

### 6. Train the Model

```bash
# Start training (from src/ directory)
python train.py --num_epochs 30 --batch_size 8

# Or with custom parameters
python train.py --num_epochs 50 --batch_size 16 --max_samples 300
```

**Training Tips:**
- Start with `--max_samples 50` for a quick test run (~5 minutes)
- Adjust `--batch_size` based on your GPU memory:
  - 4GB GPU: batch_size 4
  - 8GB GPU: batch_size 8
  - 16GB+ GPU: batch_size 16 or higher
- Monitor GPU usage: `nvidia-smi` (on another terminal)

### 7. Evaluate and Visualize

```bash
# Evaluate on validation set
python evaluate.py --model_path ../outputs/best_model.pth --save_results

# Generate prediction visualizations
python inference.py --data_dir ../data/stage1_train --num_samples 10

# Open Jupyter notebook for interactive demo
cd ../notebooks
jupyter notebook demo.ipynb
```

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size: `--batch_size 4`
- Reduce image size: modify `img_size` in config (not recommended)
- Use CPU: Model will automatically fall back to CPU

### "Data directory not found"
- Ensure dataset is downloaded to `data/stage1_train/`
- Check that image folders exist in `data/stage1_train/`

### "No module named 'albumentations'"
- Reinstall requirements: `pip install -r requirements.txt`

### Slow training on CPU
- This is expected. Consider using Google Colab (free GPU):
  1. Upload project to Google Drive
  2. Open Colab notebook
  3. Mount Drive and run training

### Import errors in notebooks
- Ensure kernel is using correct environment
- Restart kernel and run all cells

## Next Steps

After successful setup:

1. **Quick test:** Train on 50 samples to verify everything works
   ```bash
   python train.py --max_samples 50 --num_epochs 5
   ```

2. **Full training:** Train on full subset
   ```bash
   python train.py --num_epochs 30
   ```

3. **Experiment:** Try different hyperparameters, augmentations, etc.

4. **Document results:** Update README with your actual metrics

5. **Push to GitHub:** Follow instructions in README

## Getting Help

- Check the main README.md for detailed documentation
- Review code comments for implementation details
- Open an issue on GitHub for bugs or questions

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Kaggle Dataset Page](https://www.kaggle.com/c/data-science-bowl-2018)
