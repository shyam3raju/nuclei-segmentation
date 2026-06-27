# Customization Guide

How to adapt this project for your own segmentation tasks.

## Table of Contents
1. [Using a Different Dataset](#using-a-different-dataset)
2. [Modifying the Architecture](#modifying-the-architecture)
3. [Changing Hyperparameters](#changing-hyperparameters)
4. [Adding Custom Augmentations](#adding-custom-augmentations)
5. [Implementing New Metrics](#implementing-new-metrics)

---

## Using a Different Dataset

### Dataset Format Requirements

The current code expects this structure:
```
data/
└── dataset_name/
    ├── image_id_1/
    │   ├── images/
    │   │   └── image_id_1.png
    │   └── masks/
    │       ├── mask1.png
    │       └── mask2.png
    └── image_id_2/
        └── ...
```

### Adapting for Your Dataset

**Option 1: Match the Expected Format**

Organize your data to match the structure above.

**Option 2: Modify `dataset.py`**

If your dataset has a different structure, modify the `NucleiDataset` class:

```python
# In src/dataset.py, modify the __getitem__ method:

def __getitem__(self, idx):
    image_id = self.image_ids[idx]
    
    # MODIFY THIS: Load your images
    image_path = self.data_dir / f"{image_id}_image.png"  # Your naming convention
    image = np.array(Image.open(image_path))
    
    # MODIFY THIS: Load your masks
    mask_path = self.data_dir / f"{image_id}_mask.png"   # Your mask location
    mask = np.array(Image.open(mask_path))
    mask = (mask > 0).astype(np.float32)  # Ensure binary
    
    # Rest remains the same
    if self.transform:
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
    
    mask = mask.unsqueeze(0)
    return image, mask
```

### Different Image Types

**Grayscale images:**
Already handled! The code converts grayscale to RGB automatically.

**Different number of channels:**
```python
# In src/model.py:
model = UNet(in_channels=1, out_channels=1)  # For grayscale input

# In src/dataset.py, remove RGB conversion:
# Delete these lines:
# if len(image.shape) == 2:
#     image = np.stack([image] * 3, axis=-1)
```

**Multi-class segmentation:**
```python
# In src/model.py:
model = UNet(in_channels=3, out_channels=num_classes)

# In src/train.py, change loss to CrossEntropyLoss:
criterion = nn.CrossEntropyLoss()

# Masks should be class indices (0, 1, 2, ...) not binary
```

---

## Modifying the Architecture

### Changing Model Size

**Smaller model (faster, less accurate):**
```python
# In src/model.py, reduce channel counts:

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)  # Was 64
        self.down1 = Down(32, 64)    # Was 128
        self.down2 = Down(64, 128)   # Was 256
        self.down3 = Down(128, 256)  # Was 512
        # ... update the rest accordingly
```

**Larger model (more accurate, slower):**
```python
# Add more down/up blocks:

self.down5 = Down(1024, 2048)
self.up0 = Up(2048, 1024)
# Update forward pass accordingly
```

### Using Pretrained Encoder

Replace the encoder with a pretrained backbone (ResNet, EfficientNet, etc.):

```python
import torchvision.models as models

class UNetWithResNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Use pretrained ResNet as encoder
        resnet = models.resnet50(pretrained=True)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        
        # Build custom decoder with skip connections
        # ... (implementation details)
```

### Adding Attention Mechanisms

Add attention gates to focus on important features:

```python
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi

# Use in Up block:
class AttentionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.attention = AttentionGate(in_channels//2, in_channels//2, in_channels//4)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.attention(x1, x2)  # Apply attention
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
```

---

## Changing Hyperparameters

### Using `config.py`

The easiest way is to modify `src/config.py`:

```python
# In src/config.py:
BATCH_SIZE = 16        # Increase for faster training
NUM_EPOCHS = 50        # Train longer
LEARNING_RATE = 5e-5   # Lower for fine-tuning
```

### Experiment with Different Optimizers

```python
# In src/train.py, replace Adam:

# SGD with momentum
optimizer = optim.SGD(
    model.parameters(),
    lr=config['learning_rate'],
    momentum=0.9,
    weight_decay=1e-4
)

# AdamW (better generalization)
optimizer = optim.AdamW(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=0.01
)
```

### Different Learning Rate Schedules

```python
# Cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config['num_epochs']
)

# Step decay
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)

# One cycle policy (fast convergence)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=len(train_loader) * config['num_epochs']
)
```

---

## Adding Custom Augmentations

### Modify Augmentation Pipeline

In `src/dataset.py`, add more transformations:

```python
def get_transforms(img_size=256, augment=True):
    if augment:
        return A.Compose([
            A.Resize(img_size, img_size),
            
            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, 
                              rotate_limit=45, p=0.5),
            
            # Color transforms (be careful with medical images!)
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            
            # Elastic deformation (realistic for cells)
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                       std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
```

### Domain-Specific Augmentations

For microscopy images:
```python
# Add stain variations
A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, 
                     val_shift_limit=20, p=0.3)

# Simulate different exposure levels
A.RandomBrightnessContrast(brightness_limit=0.3, 
                           contrast_limit=0.3, p=0.5)
```

---

## Implementing New Metrics

### Add Custom Metric

In `src/evaluate.py`, add your metric:

```python
def precision_score(predictions, targets, threshold=0.5, smooth=1.0):
    """
    Precision = True Positives / (True Positives + False Positives)
    """
    predictions = (predictions > threshold).float()
    targets = (targets > threshold).float()
    
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    tp = (predictions * targets).sum()
    fp = (predictions * (1 - targets)).sum()
    
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision.item()

def recall_score(predictions, targets, threshold=0.5, smooth=1.0):
    """
    Recall = True Positives / (True Positives + False Negatives)
    """
    predictions = (predictions > threshold).float()
    targets = (targets > threshold).float()
    
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    tp = (predictions * targets).sum()
    fn = ((1 - predictions) * targets).sum()
    
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall.item()
```

### Use in Evaluation Loop

```python
# In evaluate_model function:
precision = precision_score(probs[i], masks[i], threshold)
recall = recall_score(probs[i], masks[i], threshold)

precision_scores.append(precision)
recall_scores.append(recall)
```

---

## Advanced Customizations

### Multi-GPU Training

```python
# In src/train.py:
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    logits = model(images)
    loss = criterion(logits, masks)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Weighted Loss for Imbalanced Data

```python
# Calculate class weights
pos_weight = (total_pixels - num_nucleus_pixels) / num_nucleus_pixels

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
```

---

## Tips for Success

1. **Start simple:** Get baseline working before adding complexity
2. **One change at a time:** Makes debugging easier
3. **Track experiments:** Use tools like TensorBoard or Weights & Biases
4. **Validate assumptions:** Check data loading, augmentations visually
5. **Monitor overfitting:** Watch train vs validation metrics
6. **Save your work:** Commit changes to git frequently

## Questions?

If you make interesting modifications, consider:
- Opening a pull request to share improvements
- Creating an issue to discuss new features
- Forking the repo for your own variant

Happy experimenting! 🚀
