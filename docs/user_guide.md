# User Guide

This comprehensive guide will walk you through using the PyTorch Lightning Template for image segmentation, from initial setup to model deployment.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Preparation](#data-preparation)
3. [Configuration](#configuration)
4. [Training Your First Model](#training-your-first-model)
5. [Monitoring Training](#monitoring-training)
6. [Validation and Testing](#validation-and-testing)
7. [Model Deployment](#model-deployment)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## Getting Started

### Prerequisites

Before you begin, ensure you have:
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended)
- At least 8GB of system RAM
- 2GB of GPU memory (minimum for small models)

### Installation

1. **Install uv (if not already installed):**
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Alternative: using pip
   pip install uv
   ```

2. **Clone the template:**
   ```bash
   git clone <repository-url>
   cd PyTorchLightning-Template
   ```

3. **Install dependencies with uv:**
   ```bash
   # Create virtual environment and install all dependencies
   uv sync
   
   # Activate the virtual environment
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch, lightning; print('Installation successful!')"
   ```

## Data Preparation

### Supported Data Formats

The template supports PNG images for both input images and segmentation masks.

### Directory Structure

Organize your data as follows:
```
data/
├── images/              # Input images
│   ├── sample_001.png
│   ├── sample_002.png
│   ├── sample_003.png
│   └── ...
└── masks/              # Ground truth segmentation masks
    ├── sample_001.png  # Must match image names
    ├── sample_002.png
    ├── sample_003.png
    └── ...
```

### Image Requirements

**Input Images:**
- Format: PNG, JPG, or other PIL-supported formats
- Channels: Grayscale (1 channel) or RGB (3 channels)
- Size: Any size (will be resized according to configuration)
- Naming: Consistent naming pattern

**Segmentation Masks:**
- Format: PNG (preferred) or other PIL-supported formats
- Type: Indexed color images with pixel values representing class IDs
- Classes: Pixel values should range from 0 to (n_classes - 1)
- Size: Must match corresponding input images
- Naming: Must exactly match input image names

### Example Data Preparation Script

```python
import os
from PIL import Image
import numpy as np

def prepare_masks(mask_dir, output_dir, class_mapping):
    """
    Convert RGB masks to indexed masks.
    
    Args:
        mask_dir: Directory containing RGB masks
        output_dir: Directory to save indexed masks
        class_mapping: Dict mapping RGB colors to class indices
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(mask_dir):
        if filename.endswith('.png'):
            mask_path = os.path.join(mask_dir, filename)
            mask = Image.open(mask_path).convert('RGB')
            mask_array = np.array(mask)
            
            # Create indexed mask
            indexed_mask = np.zeros(mask_array.shape[:2], dtype=np.uint8)
            
            for color, class_id in class_mapping.items():
                # Find pixels matching this color
                matches = np.all(mask_array == color, axis=2)
                indexed_mask[matches] = class_id
            
            # Save indexed mask
            indexed_image = Image.fromarray(indexed_mask, mode='P')
            output_path = os.path.join(output_dir, filename)
            indexed_image.save(output_path)

# Example usage
class_mapping = {
    (0, 0, 0): 0,        # Background - black
    (255, 0, 0): 1,      # Class 1 - red
    (0, 255, 0): 2,      # Class 2 - green
    (0, 0, 255): 3,      # Class 3 - blue
}

prepare_masks('raw_masks/', 'data/masks/', class_mapping)
```

## Configuration

### Understanding config.toml

The `config.toml` file controls all aspects of training. Here's a detailed breakdown:

#### Basic Configuration

```toml
[Version]
version = "0.0.1"

[DataModule]
# Data paths
image_path = "./data/images"
mask_path = "./data/masks"

# Dataset parameters
n_class = 4                    # Number of classes (including background)
image_shape = [256, 256, 1]    # [height, width, channels]
batch_size = 16                # Adjust based on GPU memory

# Data splitting
split_ratio = [0.7, 0.15, 0.15]  # [train, test, validation]
shuffle = true                   # Shuffle data before splitting

# Data augmentation
augmentation = true             # Enable/disable augmentations

# K-fold validation (optional)
use_kfold = false              # Set to true for k-fold validation
k_fold = 5                     # Number of folds
k_fold_validation_ratio = 0.2  # Validation split within each fold

[NetModule]
# Model identification
model_name = "my_segmentation_model"

# Training parameters
epochs = 200                   # Maximum training epochs
lr = 0.001                    # Learning rate

# Model architecture
unet_channels = [64, 96, 128, 196]  # Network channel progression

# Directories
checkpoint_dir = "./logs/checkpoints/"
log_dir = "./logs/"

# Early stopping
early_stopping_patience = 10          # Stop after N epochs without improvement
early_stopping_monitor = "val_loss"   # Metric to monitor
early_stopping_mode = "min"           # 'min' for loss, 'max' for accuracy
```

### Configuration for Different Use Cases

#### Medical Imaging
```toml
[DataModule]
image_path = "./medical_data/images"
mask_path = "./medical_data/masks"
n_class = 3                    # Background + 2 anatomical structures
image_shape = [512, 512, 1]    # High resolution grayscale
batch_size = 4                 # Larger images = smaller batch

[NetModule]
model_name = "medical_segmentation"
lr = 0.0001                    # Lower learning rate for medical data
epochs = 300                   # More epochs for convergence
```

#### Satellite Imagery
```toml
[DataModule]
image_path = "./satellite_data/images"
mask_path = "./satellite_data/masks"
n_class = 6                    # Multiple land use classes
image_shape = [256, 256, 3]    # RGB satellite images
batch_size = 32                # Smaller images = larger batch

[NetModule]
model_name = "satellite_segmentation"
lr = 0.002                     # Higher learning rate for RGB data
```

#### Automotive (Road Segmentation)
```toml
[DataModule]
image_path = "./automotive_data/images"
mask_path = "./automotive_data/masks"
n_class = 8                    # Road, sidewalk, car, pedestrian, etc.
image_shape = [384, 216, 3]    # 16:9 aspect ratio
batch_size = 16

[NetModule]
model_name = "road_segmentation"
lr = 0.001
early_stopping_patience = 15  # More patience for complex scenes
```

## Training Your First Model

### Step 1: Prepare Configuration

1. Edit `config.toml` with your data paths and parameters:
   ```toml
   [DataModule]
   image_path = "./data/images"
   mask_path = "./data/masks"
   n_class = 4                 # Adjust to your number of classes
   image_shape = [256, 256, 1] # Adjust to your image format
   batch_size = 16             # Adjust based on GPU memory
   ```

2. Verify your data structure:
   ```bash
   python -c "
   from Utils.utils import listFiles
   images = listFiles('./data/images', '*.png')
   masks = listFiles('./data/masks', '*.png')
   print(f'Found {len(images)} images and {len(masks)} masks')
   assert len(images) == len(masks), 'Mismatch in image and mask counts'
   print('Data verification passed!')
   "
   ```

### Step 2: Start Training

1. **Basic training:**
   ```bash
   python TrainerFit.py
   ```

2. **Monitor GPU usage (optional):**
   ```bash
   # In another terminal
   nvidia-smi -l 1
   ```

### Step 3: Understanding Training Output

During training, you'll see output like:
```
Epoch 5/200:  25%|██▌       | 5/20 [00:30<01:30, 6.00s/it, loss=0.45, v_num=1, val_loss=0.52, val_iou=0.73]
```

Key metrics:
- **loss**: Training loss (lower is better)
- **val_loss**: Validation loss (lower is better)
- **val_iou**: Validation IoU score (higher is better, max 1.0)

### Step 4: Training Completion

Training completes when:
- Maximum epochs are reached
- Early stopping is triggered (no improvement for N epochs)
- Manual interruption (Ctrl+C)

## Monitoring Training

### TensorBoard Integration

1. **Start TensorBoard:**
   ```bash
   tensorboard --logdir ./logs
   ```

2. **Open browser to:** `http://localhost:6006`

3. **Available visualizations:**
   - Loss curves (training and validation)
   - Learning rate scheduling
   - Model architecture graph
   - Hyperparameter tracking

### Key Metrics to Monitor

#### Training Metrics
- **train_loss**: Should decrease steadily
- **lr**: Learning rate (shows scheduler behavior)

#### Validation Metrics
- **val_loss**: Should decrease and track training loss
- **val_iou**: Intersection over Union (segmentation quality)

#### Warning Signs
- **Overfitting**: Training loss decreases but validation loss increases
- **Underfitting**: Both losses plateau at high values
- **Learning rate too high**: Loss fluctuates wildly
- **Learning rate too low**: Very slow convergence

### Checkpointing

- **Best model**: Automatically saved based on validation loss
- **Location**: `./logs/model_name/`
- **Filename format**: `model_name-fold=X-epoch=XXX-val_loss=X.XXXXX.ckpt`

## Validation and Testing

### Automatic Validation

Validation runs automatically during training at the end of each epoch.

### Manual Validation

Run validation on the best checkpoint:

```bash
python PredictionVal.py
```

This will:
1. Find the best checkpoint automatically
2. Load the trained model
3. Run inference on validation set
4. Calculate comprehensive metrics
5. Save results to `validation_results.json`

### Understanding Validation Results

```bash
python load_validation_results.py
```

Sample output:
```json
{
    "timestamp": "2024-01-15 14:30:22",
    "model_info": {
        "checkpoint": "./logs/model/best_model.ckpt",
        "epoch": 45,
        "val_loss": 0.234
    },
    "metrics": {
        "mean_iou": 0.782,
        "pixel_accuracy": 0.915,
        "class_iou": [0.934, 0.756, 0.698, 0.741],
        "dice_score": 0.867
    },
    "per_class_metrics": {
        "class_0": {"iou": 0.934, "dice": 0.965, "precision": 0.943, "recall": 0.987},
        "class_1": {"iou": 0.756, "dice": 0.861, "precision": 0.834, "recall": 0.890},
        "class_2": {"iou": 0.698, "dice": 0.823, "precision": 0.791, "recall": 0.857},
        "class_3": {"iou": 0.741, "dice": 0.851, "precision": 0.823, "recall": 0.881}
    }
}
```

### K-Fold Cross Validation

For more robust validation:

1. **Enable k-fold in config:**
   ```toml
   [DataModule]
   use_kfold = true
   k_fold = 5
   ```

2. **Run k-fold training:**
   ```bash
   python TrainerFitKFold.py
   ```

3. **Results**: Each fold saves a separate checkpoint

## Model Deployment

### ONNX Export

Convert your trained model for deployment:

```bash
python ModelDeploy.py
```

This creates:
- `./deployed_model/model.onnx`: Optimized ONNX model
- Verification output showing inference correctness

### Using Deployed Model

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load ONNX model
session = ort.InferenceSession("./deployed_model/model.onnx")

# Prepare input
image = Image.open("test_image.png").convert('L')  # Grayscale
image = image.resize((288, 480))  # Resize to model input size
input_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
input_array = input_array.reshape(1, 1, 480, 288)  # Add batch and channel dims

# Run inference
output = session.run(None, {"input": input_array})
predictions = output[0]  # Shape: (1, n_classes, height, width)

# Get class predictions
predicted_classes = np.argmax(predictions, axis=1)  # Shape: (1, height, width)
```

### Deployment Options

#### Edge Deployment
- Use ONNX Runtime for cross-platform inference
- Consider model quantization for smaller size
- Test on target hardware before deployment

#### Cloud Deployment
- Wrap ONNX model in REST API (FastAPI, Flask)
- Use container deployment (Docker)
- Consider batch processing for efficiency

#### Mobile Deployment
- Convert ONNX to TensorFlow Lite or Core ML
- Apply aggressive quantization
- Test memory and compute constraints

## Advanced Usage

### Custom Loss Functions

Modify `NetModule.py` to use different loss combinations:

```python
def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.forward(x)
    y_hat_s = F.softmax(y_hat, dim=1)
    
    # Custom loss combination
    ce_loss = F.cross_entropy(y_hat, y)
    dice_loss = dice.DiceLoss(mode='multiclass')(y_hat_s, y)
    focal_loss = focal.FocalLoss(alpha=0.25, gamma=2.0)(y_hat, y)
    
    # Weighted combination
    total_loss = 0.4 * ce_loss + 0.4 * dice_loss + 0.2 * focal_loss
    
    self.log("train_loss", total_loss, on_epoch=True, prog_bar=True)
    return {'loss': total_loss}
```

### Class Weighting

For imbalanced datasets, add class weights:

```python
# In NetModule.__init__()
class_weights = torch.FloatTensor([1.0, 2.0, 3.0, 1.5])  # Adjust per class
self.class_weights = class_weights

# In training_step()
ce_loss = F.cross_entropy(y_hat, y, weight=self.class_weights)
```

### Custom Data Augmentation

Modify `DataPreprocessing.py` to add custom augmentations:

```python
from torchvision import transforms
from Utils.DataAugmentation import CustomAugmentation

self.transform = transforms.Compose([
    GrayJitter(brightness=0.2, contrast=0.2),
    RandomFlip(axis=1),
    RandomRotation(degrees=15),  # Add rotation
    RandomElasticDeform(),       # Add elastic deformation
    RandomCrop2D(out_size),
    Normalize(),
    ToTensor()
])
```

### Multi-GPU Training

For faster training on multiple GPUs:

```python
# In TrainerFit.py
trainer = L.Trainer(
    accelerator="gpu",
    devices=[0, 1, 2, 3],    # Use 4 GPUs
    strategy="ddp",          # Distributed Data Parallel
    max_epochs=500,
    precision="16-mixed"     # Mixed precision for speed
)
```

### Learning Rate Scheduling

Custom learning rate schedules:

```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    
    # Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-6
    )
    
    # Or step-based decay
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.1
    )
    
    return [optimizer], [scheduler]
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size in `config.toml`
2. Reduce image size
3. Use gradient accumulation:
   ```python
   trainer = L.Trainer(accumulate_grad_batches=4)
   ```

#### Data Loading Errors
**Error**: `FileNotFoundError` or empty datasets

**Solutions**:
1. Check data paths in `config.toml`
2. Verify file permissions
3. Ensure image and mask names match exactly
4. Check file formats are supported

#### Training Not Converging
**Symptoms**: Loss stays high or fluctuates

**Solutions**:
1. Lower learning rate (try 0.0001)
2. Check data quality and labels
3. Increase batch size if possible
4. Verify loss function is appropriate

#### Validation Loss Increasing
**Symptoms**: Overfitting (val_loss > train_loss and increasing)

**Solutions**:
1. Enable data augmentation
2. Reduce model complexity
3. Add dropout layers
4. Use early stopping (already enabled)

### Performance Issues

#### Slow Training
**Solutions**:
1. Use SSD for data storage
2. Increase `num_workers` in DataLoaders
3. Use mixed precision training
4. Profile code to find bottlenecks

#### Poor Segmentation Quality
**Solutions**:
1. Check mask quality and consistency
2. Verify class distribution is balanced
3. Use appropriate loss functions (Dice for small objects)
4. Increase model capacity or training time

### Debugging Tips

#### Enable Detailed Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Visualize Data Pipeline
```python
# Add to DataPreprocessing.py
import matplotlib.pyplot as plt

def visualize_sample(self, idx):
    img, mask = self[idx]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img[0], cmap='gray')
    ax1.set_title('Image')
    ax2.imshow(mask, cmap='tab10')
    ax2.set_title('Mask')
    plt.show()
```

#### Check Model Output
```python
# Test model forward pass
model = NetModule(config)
dummy_input = torch.randn(1, 1, 256, 256)
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # Should be (1, n_classes, 256, 256)
```

## Best Practices

### Data Management
1. **Consistent naming**: Use consistent file naming patterns
2. **Quality control**: Manually inspect a subset of masks
3. **Version control**: Track different versions of your dataset
4. **Backup**: Keep backups of processed data

### Training Strategy
1. **Start small**: Begin with a subset of data to verify pipeline
2. **Baseline first**: Train a simple model before adding complexity
3. **Monitor closely**: Watch for overfitting and learning issues
4. **Document experiments**: Keep track of configuration changes

### Model Development
1. **Incremental changes**: Make one change at a time
2. **Version checkpoints**: Save models at different stages
3. **Test thoroughly**: Validate on unseen data
4. **Deploy incrementally**: Test deployment pipeline early

### Performance Optimization
1. **Profile first**: Identify bottlenecks before optimizing
2. **Batch size**: Find optimal batch size for your hardware
3. **Mixed precision**: Use automatic mixed precision for speed
4. **Data loading**: Optimize data pipeline with multiple workers

### Production Deployment
1. **Version control**: Track model versions in production
2. **Monitoring**: Monitor model performance in production
3. **Rollback plan**: Have a plan to rollback problematic models
4. **Testing**: Thoroughly test deployment pipeline

This completes the comprehensive user guide. The template is now fully documented with installation instructions, configuration options, training procedures, validation methods, deployment strategies, and troubleshooting guides.