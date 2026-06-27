# Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites

- Python 3.8+
- 5GB free disk space

## Installation (3 minutes)

```bash
# 1. Clone repo
git clone https://github.com/yourusername/nuclei-segmentation-unet.git
cd nuclei-segmentation-unet

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset (Option A - Kaggle API)
# First: Get kaggle.json from https://www.kaggle.com/account
# Then place it in ~/.kaggle/ (Linux/Mac) or C:\Users\<you>\.kaggle\ (Windows)
cd data
python download_data.py
cd ..
```

## Verify Setup (1 minute)

```bash
cd src
python test_setup.py
```

If all tests pass, you're ready!

## Train Model (30-60 minutes on GPU, 2-3 hours on CPU)

```bash
# Quick test (5 minutes)
python train.py --max_samples 50 --num_epochs 5

# Full training (recommended)
python train.py --num_epochs 30 --batch_size 8
```

## View Results (1 minute)

```bash
# Evaluate metrics
python evaluate.py --model_path ../outputs/best_model.pth --save_results

# Generate predictions
python inference.py --data_dir ../data/stage1_train --num_samples 10

# Open interactive demo
cd ../notebooks
jupyter notebook demo.ipynb
```

## Expected Results

After training for 30 epochs on ~500 images:

- **Dice Coefficient:** 0.75-0.85 (higher is better)
- **IoU Score:** 0.65-0.75
- **Training time:** ~1 hour on GPU, ~2-3 hours on CPU

## Troubleshooting

### "CUDA out of memory"
```bash
python train.py --batch_size 4  # Reduce batch size
```

### "Dataset not found"
Make sure you ran `python download_data.py` in the `data/` directory.

### "Module not found"
```bash
pip install -r requirements.txt
```

## What's Next?

- **Read README.md** for detailed documentation
- **Explore notebooks/demo.ipynb** for visualizations
- **Modify src/train.py** to experiment with hyperparameters
- **Check SETUP.md** for advanced configuration

## Key Files

- `src/train.py` - Train the model
- `src/evaluate.py` - Compute metrics
- `src/inference.py` - Run predictions
- `src/model.py` - U-Net architecture
- `notebooks/demo.ipynb` - Interactive demo

## Questions?

Check the main README.md or open an issue on GitHub.

---

**Happy segmenting! 🔬🧬**
