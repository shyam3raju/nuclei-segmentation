"""
Inference script for running predictions on new images.

This script:
1. Loads a trained model
2. Runs inference on sample images
3. Visualizes predictions overlaid on original images
4. Saves results

Useful for:
- Demo purposes
- Testing on new data
- Creating visualizations for presentations
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from model import UNet
from dataset import get_transforms


def load_model(model_path, device):
    """
    Load trained U-Net model from checkpoint.
    
    Args:
        model_path (str): Path to model checkpoint
        device: torch device
    
    Returns:
        model: Loaded U-Net model in eval mode
    """
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def predict_image(model, image_path, transform, device, threshold=0.5):
    """
    Run inference on a single image.
    
    Args:
        model: Trained U-Net model
        image_path: Path to input image
        transform: Preprocessing transforms
        device: torch device
        threshold: Threshold for binary prediction
    
    Returns:
        original_image: Original image as numpy array
        prediction_mask: Predicted binary mask
        prediction_prob: Prediction probability map
    """
    # Load image
    original_image = np.array(Image.open(image_path))
    
    # Handle grayscale images
    if len(original_image.shape) == 2:
        original_image = np.stack([original_image] * 3, axis=-1)
    
    # Apply transforms
    transformed = transform(image=original_image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
    
    # Convert to numpy
    prediction_prob = probs.squeeze().cpu().numpy()
    prediction_mask = (prediction_prob > threshold).astype(np.uint8)
    
    return original_image, prediction_mask, prediction_prob


def create_overlay(image, mask, alpha=0.5, color=(0, 255, 0)):
    """
    Create overlay visualization of mask on image.
    
    Args:
        image: Original image (H, W, 3)
        mask: Binary mask (H, W)
        alpha: Transparency for overlay
        color: RGB color for mask overlay
    
    Returns:
        overlay: Image with mask overlay
    """
    # Resize mask to match image size if needed
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    
    # Add contours for better visibility
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = cv2.drawContours(overlay.copy(), contours, -1, color, 2)
    
    return overlay


def visualize_prediction(original_image, ground_truth, prediction, 
                        save_path=None, show=True):
    """
    Create side-by-side visualization of input, ground truth, and prediction.
    
    Args:
        original_image: Original input image
        ground_truth: Ground truth mask (can be None)
        prediction: Predicted mask
        save_path: Path to save visualization
        show: Whether to display the plot
    """
    if ground_truth is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['Original Image', 'Ground Truth', 'Prediction']
        images = [original_image, ground_truth, prediction]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        titles = ['Original Image', 'Prediction']
        images = [original_image, prediction]
        axes = [axes[0], axes[1]]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def run_inference_on_folder(model, data_dir, output_dir, transform, device, 
                            num_samples=10, threshold=0.5):
    """
    Run inference on multiple images from a folder and save visualizations.
    
    Args:
        model: Trained model
        data_dir: Directory containing image folders
        output_dir: Directory to save results
        transform: Preprocessing transforms
        device: torch device
        num_samples: Number of samples to process
        threshold: Threshold for binary prediction
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of image folders
    image_folders = [d for d in data_path.iterdir() if d.is_dir()][:num_samples]
    
    print(f"Processing {len(image_folders)} images...")
    
    for idx, folder in enumerate(image_folders):
        image_id = folder.name
        image_path = folder / 'images' / f'{image_id}.png'
        
        if not image_path.exists():
            print(f"⚠ Image not found: {image_path}")
            continue
        
        # Load ground truth if available
        masks_dir = folder / 'masks'
        if masks_dir.exists():
            ground_truth = np.zeros((256, 256), dtype=np.float32)
            for mask_file in masks_dir.glob('*.png'):
                mask = np.array(Image.open(mask_file))
                ground_truth = np.maximum(ground_truth, mask)
            ground_truth = (ground_truth > 0).astype(np.uint8)
        else:
            ground_truth = None
        
        # Run prediction
        original_image, prediction_mask, prediction_prob = predict_image(
            model, image_path, transform, device, threshold
        )
        
        # Resize ground truth to match prediction if needed
        if ground_truth is not None and ground_truth.shape != prediction_mask.shape:
            ground_truth = cv2.resize(ground_truth, prediction_mask.shape[::-1], 
                                     interpolation=cv2.INTER_NEAREST)
        
        # Create visualization
        save_path = output_path / f'prediction_{idx+1}_{image_id[:8]}.png'
        visualize_prediction(
            original_image, 
            ground_truth, 
            prediction_mask,
            save_path=save_path,
            show=False
        )
        
        # Also save overlay
        overlay = create_overlay(original_image, prediction_mask)
        overlay_path = output_path / f'overlay_{idx+1}_{image_id[:8]}.png'
        Image.fromarray(overlay).save(overlay_path)
        
        print(f"✓ Processed {idx+1}/{len(image_folders)}: {image_id[:8]}...")
    
    print(f"\n✓ All predictions saved to {output_path}")


def main(args):
    """
    Main inference function.
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = load_model(args.model_path, device)
    print("✓ Model loaded successfully")
    
    # Get transforms (no augmentation for inference)
    transform = get_transforms(img_size=args.img_size, augment=False)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.image_path:
        # Single image inference
        print(f"\nRunning inference on {args.image_path}...")
        original_image, prediction_mask, prediction_prob = predict_image(
            model, args.image_path, transform, device, args.threshold
        )
        
        # Save results
        save_path = output_dir / 'prediction.png'
        visualize_prediction(
            original_image,
            None,  # No ground truth for single image
            prediction_mask,
            save_path=save_path,
            show=not args.no_display
        )
        
        # Save overlay
        overlay = create_overlay(original_image, prediction_mask)
        overlay_path = output_dir / 'overlay.png'
        Image.fromarray(overlay).save(overlay_path)
        print(f"✓ Saved overlay to {overlay_path}")
        
    elif args.data_dir:
        # Batch inference on folder
        print(f"\nRunning inference on images from {args.data_dir}...")
        run_inference_on_folder(
            model,
            args.data_dir,
            output_dir,
            transform,
            device,
            num_samples=args.num_samples,
            threshold=args.threshold
        )
    else:
        print("Error: Please provide either --image_path or --data_dir")
        return
    
    print("\n✓ Inference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with trained U-Net model')
    
    # Model path
    parser.add_argument(
        '--model_path',
        type=str,
        default='../outputs/best_model.pth',
        help='Path to trained model checkpoint'
    )
    
    # Input options
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--image_path',
        type=str,
        help='Path to single image for inference'
    )
    group.add_argument(
        '--data_dir',
        type=str,
        default='../data/stage1_train',
        help='Path to data directory for batch inference'
    )
    
    # Output options
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../outputs/predictions',
        help='Directory to save predictions'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='Number of samples to process (for batch inference)'
    )
    
    # Inference parameters
    parser.add_argument(
        '--img_size',
        type=int,
        default=256,
        help='Image size for inference'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for binary prediction'
    )
    parser.add_argument(
        '--no_display',
        action='store_true',
        help='Do not display predictions (only save)'
    )
    
    args = parser.parse_args()
    
    # Run inference
    main(args)
