"""
Script to regenerate training curves from your training results.
Based on the training output from your 5-epoch quick test.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Training data from your actual training run (5 epochs on 50 images)
epochs = [1, 2, 3, 4, 5]

# Training metrics
train_loss = [0.6758, 0.5399, 0.5019, 0.4561, 0.4406]
train_dice = [0.3696, 0.6567, 0.6605, 0.7378, 0.7462]

# Validation metrics
val_loss = [0.7577, 0.7281, 0.7229, 0.6787, 0.5740]
val_dice = [0.0013, 0.0037, 0.3810, 0.5126, 0.5950]

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Loss curves
axes[0].plot(epochs, train_loss, 'o-', linewidth=2, markersize=8, 
             label='Training Loss', color='#2E86AB')
axes[0].plot(epochs, val_loss, 's-', linewidth=2, markersize=8, 
             label='Validation Loss', color='#A23B72')
axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Loss', fontsize=13, fontweight='bold')
axes[0].set_title('Training and Validation Loss', fontsize=15, fontweight='bold')
axes[0].legend(fontsize=11, loc='upper right')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].set_xticks(epochs)

# Add value labels on points
for i, (tl, vl) in enumerate(zip(train_loss, val_loss)):
    if i == len(epochs) - 1:  # Label last point
        axes[0].text(epochs[i], tl, f'{tl:.3f}', 
                    fontsize=9, ha='right', va='bottom')
        axes[0].text(epochs[i], vl, f'{vl:.3f}', 
                    fontsize=9, ha='right', va='top')

# Plot 2: Dice coefficient curves
axes[1].plot(epochs, train_dice, 'o-', linewidth=2, markersize=8, 
             label='Training Dice', color='#2E86AB')
axes[1].plot(epochs, val_dice, 's-', linewidth=2, markersize=8, 
             label='Validation Dice', color='#A23B72')
axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Dice Coefficient', fontsize=13, fontweight='bold')
axes[1].set_title('Training and Validation Dice Coefficient', fontsize=15, fontweight='bold')
axes[1].legend(fontsize=11, loc='lower right')
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].set_xticks(epochs)
axes[1].set_ylim([0, 1])

# Add value labels on points
for i, (td, vd) in enumerate(zip(train_dice, val_dice)):
    if i == len(epochs) - 1:  # Label last point
        axes[1].text(epochs[i], td, f'{td:.3f}', 
                    fontsize=9, ha='right', va='bottom')
        axes[1].text(epochs[i], vd, f'{vd:.3f}', 
                    fontsize=9, ha='right', va='top')

# Add text box with final results
textstr = f'Final Results:\nTrain Dice: {train_dice[-1]:.4f}\nVal Dice: {val_dice[-1]:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
axes[1].text(0.05, 0.50, textstr, transform=axes[1].transAxes, fontsize=10,
            verticalalignment='center', bbox=props)

plt.tight_layout()

# Save figure
output_path = Path(__file__).parent.parent / 'outputs' / 'training_curves.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Training curves saved to: {output_path}")

# Also save high-res version
output_path_hires = Path(__file__).parent.parent / 'outputs' / 'training_curves_hires.png'
plt.savefig(output_path_hires, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ High-res version saved to: {output_path_hires}")

plt.show()

print("\nTraining Summary:")
print("="*60)
print(f"Epochs: {len(epochs)}")
print(f"Starting - Train Loss: {train_loss[0]:.4f}, Train Dice: {train_dice[0]:.4f}")
print(f"Final    - Train Loss: {train_loss[-1]:.4f}, Train Dice: {train_dice[-1]:.4f}")
print(f"Final    - Val Loss: {val_loss[-1]:.4f}, Val Dice: {val_dice[-1]:.4f}")
print(f"\nImprovement:")
print(f"  Train Dice: +{train_dice[-1] - train_dice[0]:.4f} ({((train_dice[-1]/train_dice[0] - 1)*100):.1f}% increase)")
print(f"  Val Dice:   +{val_dice[-1] - val_dice[0]:.4f}")
print("="*60)
