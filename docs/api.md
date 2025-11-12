# API Reference

This document provides detailed information about all classes, functions, and modules in the PyTorch Lightning Template.

## Table of Contents

- [Core Modules](#core-modules)
  - [DataModule](#datamodule)
  - [NetModule](#netmodule)
  - [Network](#network)
  - [DataPreprocessing](#datapreprocessing)
- [Utilities](#utilities)
  - [Utils](#utils)
  - [DataAugmentation](#dataaugmentation)
  - [Network Modules](#network-modules)
- [Loss Functions](#loss-functions)
- [Training Scripts](#training-scripts)
- [Deployment](#deployment)

## Core Modules

### DataModule

#### `DataModel`

**Class**: `lightning.LightningDataModule`

Lightning DataModule for handling data loading, splitting, and preprocessing for image segmentation tasks.

**Parameters:**
- `config` (dict): Configuration dictionary containing DataModule settings

**Key Attributes:**
- `image_path` (str): Path to input images directory
- `mask_path` (str): Path to ground truth masks directory
- `batch_size` (int): Batch size for data loaders
- `img_shape` (list): Image dimensions [height, width, channels]
- `shuffle` (bool): Whether to shuffle training data
- `split_ratio` (list): Data split ratios [train, test, validation]

**Methods:**

##### `prepare_data()`
Prepare data for training (not implemented in base version).

##### `setup(stage: Optional[str] = None)`
Set up datasets for training, validation, and testing.

**Parameters:**
- `stage` (str, optional): Current stage ('fit', 'validate', 'test', or None)

**Behavior:**
- Lists all image and mask files
- Splits data according to `split_ratio`
- Creates dataset instances for each split
- Prints dataset sizes

##### `train_dataloader()`
**Returns:** `DataLoader` for training data with shuffling enabled

##### `val_dataloader()`
**Returns:** `DataLoader` for validation data without shuffling

##### `test_dataloader()`
**Returns:** `DataLoader` for test data without shuffling

##### `teardown(stage: Optional[str] = None)`
Clean up after training (not implemented in base version).

---

### NetModule

#### `NetModule`

**Class**: `lightning.LightningModule`

Lightning Module containing the neural network architecture and training logic.

**Parameters:**
- `config` (dict): Configuration dictionary containing NetModule settings

**Key Attributes:**
- `input_size` (tuple): Input image dimensions (height, width)
- `img_chn` (int): Number of input channels
- `n_class` (int): Number of output classes
- `model_name` (str): Model name for logging and checkpoints
- `log_dir` (str): Directory for saving logs
- `out` (CNNNet): The main network architecture

**Methods:**

##### `forward(x)`
Forward pass through the network.

**Parameters:**
- `x` (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

**Returns:**
- `torch.Tensor`: Output logits of shape (batch_size, n_classes, height, width)

##### `training_step(batch, batch_idx)`
Training step for one batch.

**Parameters:**
- `batch` (tuple): Batch containing (images, masks)
- `batch_idx` (int): Batch index

**Returns:**
- `dict`: Dictionary containing the loss value

**Loss Function:**
- Combines Cross Entropy and Dice Loss
- `loss = CrossEntropy(logits, targets) + DiceLoss(softmax(logits), targets)`

##### `validation_step(batch, batch_idx)`
Validation step for one batch.

**Parameters:**
- `batch` (tuple): Batch containing (images, masks)
- `batch_idx` (int): Batch index

**Metrics Logged:**
- `val_loss`: Validation loss (CrossEntropy + Dice)
- `val_iou`: Intersection over Union score

##### `configure_optimizers()`
Configure optimizers and learning rate schedulers.

**Returns:**
- `list`: List containing optimizer and scheduler configurations

**Configuration:**
- **Optimizer**: Adam with learning rate 0.002
- **Scheduler**: ReduceLROnPlateau (factor=0.1, patience=3, min_lr=1e-8)

##### `configure_callbacks()`
Configure training callbacks.

**Returns:**
- `list`: List of callback instances

**Callbacks:**
- **EarlyStopping**: Monitors validation loss (patience=10)
- **ModelCheckpoint**: Saves best model based on validation loss
- **LearningRateMonitor**: Logs learning rate changes

##### `configure_loggers()`
Configure training loggers.

**Returns:**
- `TensorBoardLogger`: TensorBoard logger instance

##### `summary()`
Print model architecture summary using torchsummary.

---

### Network

#### `CNNNet`

**Class**: `torch.nn.Module`

Custom CNN architecture based on ResNet blocks with U-Net structure for semantic segmentation.

**Parameters:**
- `in_channels` (int): Number of input channels
- `out_channels` (int): Number of output classes
- `out_activation` (str or None): Output activation function

**Architecture Overview:**
- **Encoder**: ResNet-based downsampling with skip connections
- **Decoder**: Upsampling with feature concatenation
- **Skip Connections**: Feature maps from encoder are concatenated with decoder features

**Network Stages:**

1. **Initial Convolution**: 7x7 conv with batch normalization
2. **Encoder Blocks**:
   - Block 1: 64 channels, 1/2 resolution
   - Block 2: 96 channels, 1/4 resolution  
   - Block 3: 128 channels, 1/8 resolution
   - Block 4: 192 channels, 1/16 resolution
3. **Decoder Blocks**: Progressive upsampling with skip connections
4. **Final Output**: 3x3 conv to output classes

**Methods:**

##### `forward(x, mask=None)`
Forward pass through the network.

**Parameters:**
- `x` (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
- `mask` (torch.Tensor, optional): Optional mask for selective processing

**Returns:**
- `torch.Tensor`: Output tensor of shape (batch_size, out_channels, height, width)

#### `ResNetBlock`

**Class**: `torch.nn.Module`

Basic ResNet block with downsampling capability.

**Parameters:**
- `in_channels` (int): Number of input channels
- `out_channels` (int): Number of output channels
- `stride` (tuple): Stride for convolution (default: (2, 2))

**Architecture:**
- 3x3 conv with ReLU and batch normalization
- 3x3 conv without activation
- 1x1 conv skip connection for dimension matching
- Residual addition

#### `ResNetBlock2`

**Class**: `torch.nn.Module`

ResNet block without downsampling (identity mapping).

**Parameters:**
- `in_channels` (int): Number of input channels
- `out_channels` (int): Number of output channels

**Architecture:**
- Pre-activation design with batch normalization before convolution
- Two 3x3 convolutions with residual connection
- Maintains spatial dimensions

---

### DataPreprocessing

#### `myDataset_img`

**Class**: `torch.utils.data.Dataset`

Custom dataset class for image segmentation tasks.

**Parameters:**
- `img_list` (list): List of image file paths
- `gt_list` (list): List of ground truth mask file paths
- `out_size` (tuple): Output image size (height, width)
- `shuffle` (bool): Whether to shuffle the file lists (default: True)

**Transforms Applied:**
1. `GrayJitter`: Gray-level augmentation
2. `RandomFlip`: Random horizontal/vertical flipping
3. `RandomCrop2D`: Random cropping to target size
4. `Normalize`: Normalize pixel values to [0, 1]
5. `ToTensor`: Convert to PyTorch tensors

**Methods:**

##### `__len__()`
**Returns:** `int` - Number of samples in the dataset

##### `__getitem__(item)`
Get a single sample from the dataset.

**Parameters:**
- `item` (int): Sample index

**Returns:**
- `tuple`: (image_tensor, mask_tensor)

#### `Normalize`

**Class**: Transform class for normalizing images.

Normalizes image pixel values from [0, 255] to [0, 1] range.

**Parameters:**
- `inplace` (bool): Whether to perform in-place normalization (default: False)

#### `ToTensor`

**Class**: Transform class for converting numpy arrays to PyTorch tensors.

Converts numpy arrays to PyTorch tensors with proper dimension ordering.

## Utilities

### Utils

#### `listFiles(folder, file_filter="**/*", recursive=True)`
List files in a directory with natural sorting.

**Parameters:**
- `folder` (str): Directory path to search
- `file_filter` (str): Glob pattern for file filtering (default: "**/*")
- `recursive` (bool): Whether to search recursively (default: True)

**Returns:**
- `list`: Naturally sorted list of file paths

**Example:**
```python
image_files = listFiles("./data/images", "*.png")
```

#### `split_list(file_list, split=(0.8, 0.2, 0), shuffle=True)`
Split a list into train/test/validation sets.

**Parameters:**
- `file_list` (list): List to split
- `split` (tuple): Split ratios (train, test, validation)
- `shuffle` (bool): Whether to shuffle before splitting

**Returns:**
- `tuple`: (train_list, test_list, valid_list)

#### `k_fold_split(file_list, fold=5)`
Create k-fold cross-validation splits.

**Parameters:**
- `file_list` (list): List of items to split
- `fold` (int): Number of folds

**Returns:**
- `tuple`: (kFold_list_file, kFold_list_idx)
  - `kFold_list_file`: List of (train, validation) file lists for each fold
  - `kFold_list_idx`: List of (train, validation) indices for each fold

#### `shuffle_lists(*file_lists)`
Shuffle multiple lists while maintaining correspondence.

**Parameters:**
- `*file_lists`: Variable number of lists to shuffle together

**Returns:**
- Shuffled lists maintaining correspondence between elements

#### `read_img_list_to_npy(file_list, color_mode, shuffle=False)`
Read a list of image files into numpy arrays.

**Parameters:**
- `file_list` (list): List of image file paths
- `color_mode` (str): Color mode ('gray', 'rgb', or 'idx')
- `shuffle` (bool): Whether to shuffle the file list

**Returns:**
- `list`: List of numpy arrays

**Color Modes:**
- `'gray'`: Grayscale images (dtype: float32)
- `'rgb'`: RGB color images (dtype: float32)
- `'idx'`: Index/label images (dtype: int64)

#### `apply_colormap(im_gray)`
Apply a colormap to grayscale segmentation masks for visualization.

**Parameters:**
- `im_gray` (numpy.ndarray): Grayscale segmentation mask

**Returns:**
- `numpy.ndarray`: RGB colored segmentation mask

### DataAugmentation

#### `GrayJitter`
Data augmentation class for gray-level jittering.

Randomly adjusts brightness, contrast, and gamma of grayscale images.

#### `RandomCrop2D`  
Random cropping augmentation for 2D images.

**Parameters:**
- `out_size` (tuple): Target output size (height, width)

#### `RandomFlip`
Random flipping augmentation.

**Parameters:**
- `axis` (int): Axis along which to flip (0 for vertical, 1 for horizontal)

### Network Modules

#### `Conv2dReLU`
Convolution block with optional batch normalization and ReLU activation.

**Parameters:**
- `in_channels` (int): Number of input channels
- `out_channels` (int): Number of output channels
- `kernel_size` (int or tuple): Convolution kernel size
- `padding` (int or tuple): Padding size
- `stride` (int or tuple): Stride size
- `use_batchnorm` (bool): Whether to use batch normalization

#### `Activation`
Configurable activation function module.

**Parameters:**
- `name` (str or None): Activation function name ('relu', 'sigmoid', 'softmax', etc.)

## Loss Functions

### DiceLoss
Implements Dice loss for segmentation tasks.

**Parameters:**
- `mode` (str): Loss mode ('multiclass', 'binary', or 'multilabel')
- `smooth` (float): Smoothing factor to avoid division by zero

### FocalLoss
Implements Focal loss for handling class imbalance.

**Parameters:**
- `alpha` (float): Weighting factor for rare class
- `gamma` (float): Focusing parameter
- `reduction` (str): Reduction method ('mean', 'sum', or 'none')

### JaccardLoss
Implements Jaccard (IoU) loss for segmentation.

**Parameters:**
- `mode` (str): Loss mode ('multiclass', 'binary', or 'multilabel')
- `smooth` (float): Smoothing factor

### LovaszLoss
Implements Lovász loss for structured prediction.

**Parameters:**
- `mode` (str): Loss mode
- `per_image` (bool): Whether to compute loss per image

## Training Scripts

### TrainerFit.py
Main training script for standard training.

**Key Components:**
- Loads configuration from `config.toml`
- Sets up data module and network module
- Configures Lightning trainer with GPU support
- Starts training process

### TrainerFitKFold.py
Training script for k-fold cross-validation.

**Features:**
- Implements k-fold cross-validation
- Trains separate models for each fold
- Aggregates results across folds

## Deployment

### ModelDeploy.py
Script for converting trained models to ONNX format for deployment.

**Features:**
- Automatically finds best checkpoint
- Converts PyTorch model to ONNX
- Verifies inference correctness
- Supports model optimization and quantization

### deploy_onnxmodel.py
Utility functions for ONNX model deployment.

#### `save_onnxmodel(checkpoint_path, out_path, opset=18, input_shape, enable_quantize=False, saved_model_filename="model.onnx")`
Save PyTorch Lightning model as ONNX format.

**Parameters:**
- `checkpoint_path` (str): Path to PyTorch Lightning checkpoint
- `out_path` (str): Output directory for ONNX model
- `opset` (int): ONNX opset version
- `input_shape` (tuple): Input tensor shape
- `enable_quantize` (bool): Whether to apply quantization
- `saved_model_filename` (str): Output filename

#### `load_onnxmodel(model_path)`
Load ONNX model for inference.

**Parameters:**
- `model_path` (str): Path to ONNX model file

**Returns:**
- Model buffer ready for ONNX Runtime

## Configuration Schema

### DataModule Configuration
```toml
[DataModule]
image_path = "./data/images"      # str: Path to input images
mask_path = "./data/masks"        # str: Path to ground truth masks
n_class = 12                      # int: Number of segmentation classes
image_shape = [480, 288, 1]       # list: Input dimensions [H, W, C]
batch_size = 16                   # int: Training batch size
split_ratio = [0.6, 0.2, 0.2]     # list: [train, test, validation] ratios
use_kfold = false                 # bool: Enable k-fold validation
k_fold = 5                        # int: Number of folds
k_fold_validation_ratio = 0.2     # float: Validation ratio within each fold
augmentation = true               # bool: Enable data augmentation
shuffle = true                    # bool: Shuffle training data
```

### NetModule Configuration
```toml
[NetModule]
model_name = "segmentation_model"        # str: Model name for logging
unet_channels = [64, 96, 128, 196]      # list: Channel configuration
epochs = 500                            # int: Maximum training epochs
lr = 0.001                              # float: Learning rate
checkpoint_dir = "./logs/checkpoints/"  # str: Checkpoint directory
log_dir = "./logs/"                     # str: Logging directory
early_stopping_patience = 7             # int: Early stopping patience
early_stopping_monitor = "val_loss"     # str: Metric to monitor
early_stopping_mode = "min"             # str: Monitor mode (min/max)
```

## Error Handling

### Common Issues and Solutions

#### CUDA Out of Memory
- Reduce `batch_size` in configuration
- Use gradient accumulation
- Enable mixed precision training

#### File Not Found Errors
- Verify `image_path` and `mask_path` in configuration
- Ensure data directory structure is correct
- Check file permissions

#### Checkpoint Loading Issues
- Verify checkpoint file exists and is not corrupted
- Check model architecture compatibility
- Ensure proper PyTorch Lightning version

## Performance Optimization

### Memory Optimization
- Use appropriate batch sizes for your GPU
- Enable gradient checkpointing for large models
- Use mixed precision training (automatic in Lightning)

### Training Speed
- Use multiple GPUs with DDP strategy
- Optimize data loading with multiple workers
- Use compiled models where possible

### Inference Optimization
- Convert models to ONNX for deployment
- Use TensorRT for NVIDIA GPUs
- Apply model quantization for edge deployment