"""
Configuration file for nuclei segmentation project.

Centralized location for all hyperparameters and settings.
Modify values here instead of scattered throughout the codebase.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = ROOT_DIR / 'data' / 'stage1_train'
OUTPUT_DIR = ROOT_DIR / 'outputs'
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# Input/output channels
IN_CHANNELS = 3      # RGB images
OUT_CHANNELS = 1     # Binary segmentation

# Image size (input will be resized to this)
IMG_SIZE = 256       # Balance between detail and computational cost

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Batch size (adjust based on GPU memory)
# 4GB GPU: 4
# 8GB GPU: 8
# 16GB+ GPU: 16 or higher
BATCH_SIZE = 8

# Number of training epochs
NUM_EPOCHS = 30      # Usually converges by 20-25

# Learning rate
LEARNING_RATE = 1e-4  # Standard for Adam optimizer

# Loss function weights
BCE_WEIGHT = 0.5      # Weight for BCE loss (Dice gets 1 - BCE_WEIGHT)

# Optimizer settings
WEIGHT_DECAY = 1e-5   # L2 regularization

# Learning rate scheduler
LR_SCHEDULER_PATIENCE = 5     # Epochs to wait before reducing LR
LR_SCHEDULER_FACTOR = 0.5     # Multiply LR by this when reducing

# ============================================================================
# DATASET
# ============================================================================

# Train/validation split
VAL_SPLIT = 0.2      # 20% for validation

# Data loading
NUM_WORKERS = 0      # Number of worker processes
                     # Set to 2-4 on Linux/Mac for faster loading
                     # Keep at 0 on Windows to avoid issues

PIN_MEMORY = True    # Faster GPU transfer

# Dataset size limit (None = use all data)
MAX_SAMPLES = None   # Set to e.g., 200 to use subset

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

# Augmentation probabilities (0.0 = never, 1.0 = always)
AUG_HORIZONTAL_FLIP_PROB = 0.5
AUG_VERTICAL_FLIP_PROB = 0.5
AUG_ROTATE90_PROB = 0.5

# Normalization (ImageNet stats)
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

# ============================================================================
# EVALUATION
# ============================================================================

# Threshold for converting probabilities to binary predictions
PREDICTION_THRESHOLD = 0.5

# Metrics to compute
COMPUTE_DICE = True
COMPUTE_IOU = True

# ============================================================================
# INFERENCE
# ============================================================================

# Number of sample predictions to generate
NUM_INFERENCE_SAMPLES = 10

# Overlay visualization settings
OVERLAY_ALPHA = 0.5          # Transparency for mask overlay
OVERLAY_COLOR = (0, 255, 0)  # Green color for predictions

# ============================================================================
# LOGGING & CHECKPOINTING
# ============================================================================

# Print training stats every N batches
PRINT_FREQ = 10

# Save checkpoint every N epochs
SAVE_FREQ = 5

# Save best model based on this metric
BEST_MODEL_METRIC = 'val_dice'  # Options: 'val_loss', 'val_dice', 'val_iou'

# Keep only best N checkpoints
MAX_CHECKPOINTS = 3

# ============================================================================
# DEVICE SETTINGS
# ============================================================================

# Device selection (auto-detect if None)
DEVICE = None  # None = auto, 'cuda' = force GPU, 'cpu' = force CPU

# Mixed precision training (faster on modern GPUs)
USE_AMP = False  # Set to True if you have a modern GPU (Volta or newer)

# ============================================================================
# EXPERIMENT SETTINGS
# ============================================================================

# Experiment name (for organizing multiple runs)
EXPERIMENT_NAME = 'baseline'

# Tags for this run
TAGS = ['unet', 'nuclei', 'segmentation']

# Notes about this experiment
NOTES = """
Baseline U-Net model with:
- Combined BCE + Dice loss
- Standard augmentations (flips, rotations)
- Adam optimizer with ReduceLROnPlateau scheduler
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config_dict():
    """
    Get configuration as a dictionary.
    Useful for logging or saving config with results.
    """
    return {
        # Paths
        'data_dir': str(DATA_DIR),
        'output_dir': str(OUTPUT_DIR),
        
        # Model
        'in_channels': IN_CHANNELS,
        'out_channels': OUT_CHANNELS,
        'img_size': IMG_SIZE,
        
        # Training
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'bce_weight': BCE_WEIGHT,
        'weight_decay': WEIGHT_DECAY,
        
        # Dataset
        'val_split': VAL_SPLIT,
        'max_samples': MAX_SAMPLES,
        'random_seed': RANDOM_SEED,
        
        # Augmentation
        'horizontal_flip': AUG_HORIZONTAL_FLIP_PROB,
        'vertical_flip': AUG_VERTICAL_FLIP_PROB,
        'rotate90': AUG_ROTATE90_PROB,
        
        # Other
        'prediction_threshold': PREDICTION_THRESHOLD,
        'experiment_name': EXPERIMENT_NAME,
    }


def print_config():
    """Print current configuration."""
    print("="*60)
    print("CONFIGURATION")
    print("="*60)
    config = get_config_dict()
    for key, value in config.items():
        print(f"{key:25} {value}")
    print("="*60)


if __name__ == "__main__":
    print_config()
