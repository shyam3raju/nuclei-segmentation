# Outputs Directory

This directory stores training outputs and predictions.

**Contents after training:**
- `best_model.pth` - Model checkpoint with best validation Dice score
- `final_model.pth` - Model checkpoint after final epoch
- `training_curves.png` - Training/validation loss and Dice curves
- `evaluation_results.txt` - Quantitative metrics on validation set
- `predictions/` - Sample prediction visualizations

**Note:** Model checkpoints (*.pth, *.pt) are excluded from git tracking to keep the repository size manageable.
