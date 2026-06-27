"""
Training script for U-Net nuclei segmentation.

This script handles:
1. Model initialization
2. Loss function and optimizer setup
3. Training loop with validation
4. Checkpointing best model
5. Logging metrics and saving training curves

Key design choices:
- Combined BCE + Dice Loss: BCE for pixel-wise accuracy, Dice for overlap quality
- Adam optimizer: Adaptive learning rates work well for segmentation
- Learning rate scheduling: Reduce LR when validation stops improving
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import UNet
from dataset import get_dataloaders


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    
    Dice coefficient measures overlap between prediction and ground truth:
        Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Where A = predicted mask, B = ground truth mask
    
    Why Dice Loss:
    - Handles class imbalance better than BCE alone
      (nuclei are small compared to background)
    - Directly optimizes the metric we care about (overlap)
    - Range: [0, 1] where 1 = perfect overlap
    
    We minimize (1 - Dice) to turn it into a loss.
    """
    
    def __init__(self, smooth=1.0):
        """
        Args:
            smooth (float): Smoothing factor to avoid division by zero
                          Also helps with gradient stability
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions (torch.Tensor): Model output after sigmoid, shape (N, 1, H, W)
            targets (torch.Tensor): Ground truth masks, shape (N, 1, H, W)
        
        Returns:
            torch.Tensor: Dice loss value (scalar)
        """
        # Flatten tensors to compute intersection and union
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Compute intersection and union
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        # Return loss (1 - dice)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss.
    
    Why combine both:
    - BCE: Good for pixel-level accuracy, well-tested gradients
    - Dice: Good for handling class imbalance and measuring overlap
    - Together: BCE handles easy pixels, Dice focuses on hard boundary cases
    
    This combination is standard practice in medical image segmentation.
    """
    
    def __init__(self, bce_weight=0.5):
        """
        Args:
            bce_weight (float): Weight for BCE loss (Dice gets weight 1 - bce_weight)
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()  # Combines Sigmoid + BCE for numerical stability
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
    
    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model output (before sigmoid)
            targets (torch.Tensor): Ground truth masks
        """
        # BCE loss on logits
        bce_loss = self.bce(logits, targets)
        
        # Dice loss on probabilities (after sigmoid)
        probs = torch.sigmoid(logits)
        dice_loss = self.dice(probs, targets)
        
        # Combine
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def dice_coefficient(predictions, targets, threshold=0.5, smooth=1.0):
    """
    Calculate Dice coefficient for evaluation.
    
    This is the metric we report to evaluate segmentation quality.
    Higher is better, range [0, 1].
    """
    predictions = (predictions > threshold).float()
    targets = (targets > threshold).float()
    
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    intersection = (predictions * targets).sum()
    dice = (2. * intersection + smooth) / (
        predictions.sum() + targets.sum() + smooth
    )
    
    return dice.item()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Returns:
        avg_loss (float): Average loss over all batches
        avg_dice (float): Average Dice coefficient over all batches
    """
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            dice = dice_coefficient(probs, masks)
        
        running_loss += loss.item()
        running_dice += dice
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}'
        })
    
    avg_loss = running_loss / len(dataloader)
    avg_dice = running_dice / len(dataloader)
    
    return avg_loss, avg_dice


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Returns:
        avg_loss (float): Average validation loss
        avg_dice (float): Average validation Dice coefficient
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            logits = model(images)
            loss = criterion(logits, masks)
            
            # Calculate metrics
            probs = torch.sigmoid(logits)
            dice = dice_coefficient(probs, masks)
            
            running_loss += loss.item()
            running_dice += dice
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}'
            })
    
    avg_loss = running_loss / len(dataloader)
    avg_dice = running_dice / len(dataloader)
    
    return avg_loss, avg_dice


def plot_training_curves(history, save_path):
    """
    Plot and save training curves (loss and Dice coefficient).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Dice curves
    axes[1].plot(history['train_dice'], label='Train Dice')
    axes[1].plot(history['val_dice'], label='Val Dice')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Coefficient')
    axes[1].set_title('Training and Validation Dice Coefficient')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def train(config):
    """
    Main training function.
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading dataset...")
    train_loader, val_loader = get_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        val_split=config['val_split'],
        max_samples=config.get('max_samples', None),
        num_workers=config['num_workers']
    )
    
    # Create model
    print("\nInitializing model...")
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    # Loss and optimizer
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler: reduce LR when validation metric plateaus
    # Why: Helps model converge to better minima when it gets stuck
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # We want to maximize Dice
        factor=0.5,
        patience=5
    )
    
    # Training loop
    print("\nStarting training...")
    best_dice = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': []
    }
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Train
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_dice = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_dice)
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            checkpoint_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"✓ Saved best model (Dice: {best_dice:.4f})")
    
    # Save final model
    final_path = output_dir / 'final_model.pth'
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to {final_path}")
    
    # Plot training curves
    plot_training_curves(history, output_dir / 'training_curves.png')
    
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"{'='*50}")
    
    return model, history


if __name__ == "__main__":
    # Configuration
    # These are hyperparameters you can tune for better performance
    config = {
        'data_dir': '../data/stage1_train',
        'output_dir': '../outputs',
        'batch_size': 8,           # Increase if you have more GPU memory
        'img_size': 256,           # Standard size balancing detail and speed
        'num_epochs': 30,          # May need more for full convergence
        'learning_rate': 1e-4,     # Adam default, works well for most cases
        'val_split': 0.2,          # 20% for validation
        'num_workers': 0,          # Set to 2-4 on Linux/Mac for faster loading
        'max_samples': None,       # Set to e.g., 200 to use subset of data
    }
    
    # Parse command line arguments (optional)
    parser = argparse.ArgumentParser(description='Train U-Net for nuclei segmentation')
    parser.add_argument('--data_dir', type=str, default=config['data_dir'])
    parser.add_argument('--output_dir', type=str, default=config['output_dir'])
    parser.add_argument('--batch_size', type=int, default=config['batch_size'])
    parser.add_argument('--num_epochs', type=int, default=config['num_epochs'])
    parser.add_argument('--lr', type=float, default=config['learning_rate'])
    parser.add_argument('--max_samples', type=int, default=None)
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config.update({
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.lr,
        'max_samples': args.max_samples
    })
    
    # Train
    model, history = train(config)
