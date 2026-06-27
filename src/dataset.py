"""
Dataset loader and preprocessing for nuclei segmentation.

This module handles:
1. Loading microscopy images and their corresponding masks
2. Preprocessing (resize, normalize)
3. Data augmentation (flips, rotations) to improve model generalization
4. Train/validation split

Why these choices:
- Resize to 256x256: Standard size that balances detail and memory/speed
- Normalization to [0,1]: Helps neural network training convergence
- Augmentation: Small dataset needs augmentation to prevent overfitting
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

class NucleiDataset(Dataset):
    """
    Dataset class for nuclei segmentation.
    
    Each sample in the dataset consists of:
    - An image of cell nuclei (RGB or grayscale)
    - A binary mask where 1 = nucleus, 0 = background
    
    The dataset handles the common format where each nucleus has a separate mask file,
    and combines them into a single binary segmentation mask.
    """
    
    def __init__(self, image_ids, data_dir, transform=None, img_size=256):
        """
        Args:
            image_ids (list): List of image folder names/IDs
            data_dir (str): Path to data directory (e.g., 'data/stage1_train')
            transform (albumentations.Compose): Augmentation pipeline
            img_size (int): Target image size (will resize to img_size x img_size)
        """
        self.image_ids = image_ids
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.img_size = img_size
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        Load and return a single image-mask pair.
        
        Returns:
            image (torch.Tensor): Shape (3, H, W), normalized to [0, 1]
            mask (torch.Tensor): Shape (1, H, W), binary values {0, 1}
        """
        image_id = self.image_ids[idx]
        image_folder = self.data_dir / image_id
        
        # Load image
        image_path = image_folder / 'images' / f'{image_id}.png'
        image = np.array(Image.open(image_path))
        
        # Handle grayscale images - convert to RGB for consistency
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Load and combine all masks for this image
        # Each nucleus may have its own mask file - we combine them into one binary mask
        masks_dir = image_folder / 'masks'
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        for mask_file in masks_dir.glob('*.png'):
            single_mask = np.array(Image.open(mask_file))
            # Combine masks: any pixel covered by any mask becomes 1
            mask = np.maximum(mask, single_mask)
        
        # Ensure mask is binary
        mask = (mask > 0).astype(np.float32)
        
        # Apply transformations (resize, augmentation, normalization)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Add channel dimension to mask: (H, W) -> (1, H, W)
        mask = mask.unsqueeze(0)
        
        return image, mask


def get_transforms(img_size=256, augment=True):
    """
    Create augmentation pipeline using Albumentations.
    
    Why these augmentations:
    - Horizontal/Vertical Flip: Nuclei have no preferred orientation
    - Rotation: Images can be captured at any angle
    - Normalize: Standard ImageNet normalization helps with transfer learning
      (though we're training from scratch here, it's still a good practice)
    
    Args:
        img_size (int): Target size for images
        augment (bool): If True, apply augmentation; if False, only resize and normalize
    
    Returns:
        albumentations.Compose: Transformation pipeline
    """
    if augment:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # Normalize to [0, 1] and convert to tensor
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        # Validation/test: only resize and normalize, no random augmentation
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


def prepare_data(data_dir, val_split=0.2, max_samples=None, random_state=42):
    """
    Prepare train and validation datasets.
    
    Args:
        data_dir (str): Path to data directory (e.g., 'data/stage1_train')
        val_split (float): Fraction of data to use for validation (0.2 = 20%)
        max_samples (int): If set, limit dataset to this many samples (for faster iteration)
        random_state (int): Random seed for reproducible splits
    
    Returns:
        train_ids (list): Image IDs for training
        val_ids (list): Image IDs for validation
    """
    data_path = Path(data_dir)
    
    # Get all image IDs (folder names in stage1_train)
    all_ids = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    # Limit dataset size if specified (useful for quick experiments)
    if max_samples and max_samples < len(all_ids):
        all_ids = all_ids[:max_samples]
    
    # Split into train and validation sets
    # Why 80/20 split: Standard practice - enough data for training while having
    # a reasonable validation set to detect overfitting
    train_ids, val_ids = train_test_split(
        all_ids,
        test_size=val_split,
        random_state=random_state
    )
    
    print(f"Dataset prepared:")
    print(f"  Total images: {len(all_ids)}")
    print(f"  Training: {len(train_ids)}")
    print(f"  Validation: {len(val_ids)}")
    
    return train_ids, val_ids


def get_dataloaders(data_dir, batch_size=8, img_size=256, val_split=0.2, 
                    max_samples=None, num_workers=0):
    """
    Create train and validation data loaders.
    
    Args:
        data_dir (str): Path to data directory
        batch_size (int): Number of samples per batch
        img_size (int): Target image size
        val_split (float): Validation split ratio
        max_samples (int): Optional limit on dataset size
        num_workers (int): Number of worker processes for data loading
    
    Returns:
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
    """
    # Prepare train/val split
    train_ids, val_ids = prepare_data(
        data_dir,
        val_split=val_split,
        max_samples=max_samples
    )
    
    # Create datasets with appropriate transforms
    train_dataset = NucleiDataset(
        train_ids,
        data_dir,
        transform=get_transforms(img_size, augment=True),
        img_size=img_size
    )
    
    val_dataset = NucleiDataset(
        val_ids,
        data_dir,
        transform=get_transforms(img_size, augment=False),
        img_size=img_size
    )
    
    # Create data loaders
    # Why shuffle=True for train: Prevents model from learning order-dependent patterns
    # Why shuffle=False for val: Order doesn't matter, and consistent order helps debugging
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer if using CUDA
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test of the dataset loader
    print("Testing dataset loader...")
    
    # Adjust this path to your data location
    data_dir = "../data/stage1_train"
    
    if not Path(data_dir).exists():
        print(f"Data directory not found: {data_dir}")
        print("Please download the dataset first using data/download_data.py")
    else:
        # Create data loaders
        train_loader, val_loader = get_dataloaders(
            data_dir,
            batch_size=4,
            max_samples=20  # Use only 20 samples for quick test
        )
        
        # Load one batch
        images, masks = next(iter(train_loader))
        
        print(f"\nBatch shapes:")
        print(f"  Images: {images.shape}")  # Should be (batch_size, 3, 256, 256)
        print(f"  Masks: {masks.shape}")    # Should be (batch_size, 1, 256, 256)
        print(f"\nValue ranges:")
        print(f"  Image min/max: {images.min():.3f} / {images.max():.3f}")
        print(f"  Mask unique values: {torch.unique(masks)}")
        print("\nDataset loader test passed! ✓")
