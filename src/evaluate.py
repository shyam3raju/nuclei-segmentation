"""
Evaluation script to compute metrics on validation set.

This script loads a trained model and computes:
1. Dice coefficient (primary metric for segmentation)
2. IoU (Intersection over Union) - another common segmentation metric

These metrics tell us how well the model segments nuclei.
"""

import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

from model import UNet
from dataset import get_dataloaders


def dice_coefficient(predictions, targets, threshold=0.5, smooth=1.0):
    """
    Calculate Dice coefficient.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Measures overlap between prediction and ground truth.
    Range: [0, 1] where 1 = perfect overlap
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


def iou_score(predictions, targets, threshold=0.5, smooth=1.0):
    """
    Calculate Intersection over Union (IoU), also called Jaccard index.
    
    IoU = |A ∩ B| / |A ∪ B|
    
    Another standard metric for segmentation.
    Range: [0, 1] where 1 = perfect overlap
    
    Relationship to Dice: Dice = 2*IoU / (1 + IoU)
    Dice is more forgiving than IoU for the same overlap.
    """
    predictions = (predictions > threshold).float()
    targets = (targets > threshold).float()
    
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained U-Net model
        dataloader: DataLoader for evaluation data
        device: torch device (cuda or cpu)
        threshold: Threshold for converting probabilities to binary mask
    
    Returns:
        dict: Dictionary with average Dice and IoU scores
    """
    model.eval()
    
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            # Calculate metrics for each image in batch
            for i in range(images.size(0)):
                dice = dice_coefficient(probs[i], masks[i], threshold)
                iou = iou_score(probs[i], masks[i], threshold)
                
                dice_scores.append(dice)
                iou_scores.append(iou)
            
            # Update progress bar
            pbar.set_postfix({
                'dice': f'{np.mean(dice_scores):.4f}',
                'iou': f'{np.mean(iou_scores):.4f}'
            })
    
    results = {
        'dice_mean': np.mean(dice_scores),
        'dice_std': np.std(dice_scores),
        'iou_mean': np.mean(iou_scores),
        'iou_std': np.std(iou_scores),
        'num_samples': len(dice_scores)
    }
    
    return results


def main(args):
    """
    Main evaluation function.
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_dice' in checkpoint:
            print(f"Validation Dice at checkpoint: {checkpoint['val_dice']:.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✓ Model loaded successfully")
    
    # Load data
    print("\nLoading validation data...")
    _, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_split=args.val_split,
        num_workers=0
    )
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(model, val_loader, device, threshold=args.threshold)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of samples: {results['num_samples']}")
    print(f"Threshold: {args.threshold}")
    print("-"*60)
    print(f"Dice Coefficient: {results['dice_mean']:.4f} ± {results['dice_std']:.4f}")
    print(f"IoU Score:        {results['iou_mean']:.4f} ± {results['iou_std']:.4f}")
    print("="*60)
    
    # Interpretation guide
    print("\nInterpretation:")
    dice_mean = results['dice_mean']
    if dice_mean >= 0.85:
        print("✓ Excellent segmentation performance!")
    elif dice_mean >= 0.75:
        print("✓ Good segmentation performance")
    elif dice_mean >= 0.65:
        print("⚠ Moderate performance - consider more training or data augmentation")
    else:
        print("⚠ Poor performance - check data preprocessing and model training")
    
    # Save results
    if args.save_results:
        output_path = Path(args.model_path).parent / 'evaluation_results.txt'
        with open(output_path, 'w') as f:
            f.write("EVALUATION RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Number of samples: {results['num_samples']}\n")
            f.write(f"Threshold: {args.threshold}\n")
            f.write("-"*60 + "\n")
            f.write(f"Dice Coefficient: {results['dice_mean']:.4f} ± {results['dice_std']:.4f}\n")
            f.write(f"IoU Score:        {results['iou_mean']:.4f} ± {results['iou_std']:.4f}\n")
            f.write("="*60 + "\n")
        print(f"\n✓ Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate U-Net model on validation set')
    
    # Model and data paths
    parser.add_argument(
        '--model_path',
        type=str,
        default='../outputs/best_model.pth',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../data/stage1_train',
        help='Path to dataset directory'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=256,
        help='Image size (will resize to img_size x img_size)'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help='Validation split ratio'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for converting probabilities to binary predictions'
    )
    parser.add_argument(
        '--save_results',
        action='store_true',
        help='Save results to text file'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    results = main(args)
