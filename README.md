# PyTorch Lightning Template for Image Segmentation

A comprehensive template for building image segmentation models using PyTorch Lightning. This template provides a complete framework for training, validation, and deployment of deep learning models with a focus on semantic segmentation tasks.

## 🌟 Features

- **PyTorch Lightning Integration**: Modern, scalable deep learning framework
- **Custom CNN Architecture**: Optimized ResNet-based U-Net architecture for segmentation
- **Comprehensive Loss Functions**: Multiple loss functions including Dice, Focal, Jaccard, and Lovász losses
- **Data Augmentation**: Built-in data augmentation pipeline
- **K-Fold Cross Validation**: Support for robust model validation
- **Model Deployment**: ONNX export for production deployment
- **Tensorboard Integration**: Real-time training monitoring
- **Configuration Management**: TOML-based configuration system
- **Multi-GPU Support**: Scalable training on multiple GPUs

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training](#training)
- [Validation](#validation)
- [Model Deployment](#model-deployment)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Examples](#examples)
- [Contributing](#contributing)

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-compatible GPU (recommended)

### 1. Install uv (if not already installed)

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: using pip
pip install uv
```

### 2. Clone and Setup Project

```bash
git clone <repository-url>
cd PyTorchLightning-Template
```

### 3. Install Dependencies with uv

```bash
# Create virtual environment and install all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 4. Verify Installation

```bash
python -c "import torch; import lightning; print(f'PyTorch: {torch.__version__}, Lightning: {lightning.__version__}')"
```

> **Note**: `uv` automatically handles PyTorch installation with CUDA support based on your system. The `uv.lock` file ensures reproducible builds across different environments.

## 🏃‍♂️ Quick Start

### 1. Prepare Your Data

Organize your data in the following structure:
```
data/
├── images/          # Input images (.png format)
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── masks/           # Ground truth masks (.png format)
    ├── image_001.png
    ├── image_002.png
    └── ...
```

### 2. Configure the Model

Edit `config.toml` to match your dataset:

```toml
[DataModule]
image_path = "./data/images"
mask_path = "./data/masks"
n_class = 12              # Number of classes in your dataset
image_shape = [480, 288, 1]  # [height, width, channels]
batch_size = 16
split_ratio = [0.6, 0.2, 0.2]  # [train, test, validation]

[NetModule]
model_name = "my_segmentation_model"
epochs = 500
lr = 0.001
```

### 3. Start Training

```bash
python TrainerFit.py
```

### 4. Monitor Training

Open Tensorboard to monitor training progress:
```bash
tensorboard --logdir ./logs
```

### 5. Validate Model

```bash
python PredictionVal.py
```

### 6. Deploy Model

Convert trained model to ONNX format for deployment:
```bash
python ModelDeploy.py
```

## ⚙️ Configuration

The template uses a TOML configuration file (`config.toml`) for easy parameter management. Here are the key configuration sections:

### DataModule Configuration

```toml
[DataModule]
image_path = "./data/images"     # Path to input images
mask_path = "./data/masks"       # Path to ground truth masks
n_class = 12                     # Number of segmentation classes
image_shape = [480, 288, 1]      # Input image dimensions [H, W, C]
batch_size = 16                  # Training batch size
split_ratio = [0.6, 0.2, 0.2]    # Train/test/validation split
use_kfold = false                # Enable k-fold cross validation
k_fold = 5                       # Number of folds
augmentation = true              # Enable data augmentation
shuffle = true                   # Shuffle training data
```

### NetModule Configuration

```toml
[NetModule]
model_name = "segmentation_model"        # Model name for logging
unet_channels = [64, 96, 128, 196]      # U-Net channel configuration
epochs = 500                            # Maximum training epochs
lr = 0.001                              # Learning rate
checkpoint_dir = "./logs/checkpoints/"  # Checkpoint directory
log_dir = "./logs/"                     # Logging directory
early_stopping_patience = 7             # Early stopping patience
early_stopping_monitor = "val_loss"     # Metric to monitor
```

## 🎯 Training

### Basic Training

```bash
python TrainerFit.py
```

### K-Fold Cross Validation

Enable k-fold validation in `config.toml`:
```toml
[DataModule]
use_kfold = true
k_fold = 5
```

Then run:
```bash
python TrainerFitKFold.py
```

### Multi-GPU Training

The trainer automatically detects and uses available GPUs. For specific GPU configuration:

```python
trainer = L.Trainer(
    accelerator="gpu",
    devices=[0, 1],  # Use specific GPUs
    strategy="ddp",  # Distributed data parallel
    max_epochs=500
)
```

### Training Monitoring

- **Tensorboard**: `tensorboard --logdir ./logs`
- **Checkpoints**: Saved in `./logs/model_name/`
- **Metrics**: Training/validation loss and IoU

## 📊 Validation

### Run Validation

```bash
python PredictionVal.py
```

This will:
- Load the best checkpoint automatically
- Run inference on validation set
- Calculate metrics (IoU, Dice score, etc.)
- Save prediction results
- Generate validation report

### Load Validation Results

```bash
python load_validation_results.py
```

### Custom Validation Metrics

The template includes various metrics:
- **IoU (Intersection over Union)**
- **Dice Score**
- **Pixel Accuracy**
- **Mean Class Accuracy**

## 🚀 Model Deployment

### Export to ONNX

```bash
python ModelDeploy.py
```

This will:
- Find the best checkpoint
- Convert model to ONNX format
- Verify inference correctness
- Save optimized model for deployment

### Use Deployed Model

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("./deployed_model/model.onnx")

# Run inference
input_data = np.random.rand(1, 1, 480, 288).astype(np.float32)
output = session.run(None, {"input": input_data})
```

## 📁 Project Structure

```
PyTorchLightning-Template/
├── config.toml                 # Configuration file
├── pyproject.toml             # Project dependencies
├── README.md                  # This file
├── 
├── # Core modules
├── DataModule.py              # Lightning DataModule
├── NetModule.py               # Lightning Module (model)
├── Network.py                 # CNN architecture
├── DataPreprocessing.py       # Dataset class and transforms
├── 
├── # Training scripts
├── TrainerFit.py              # Basic training
├── TrainerFitKFold.py         # K-fold training
├── 
├── # Validation and deployment
├── PredictionVal.py           # Model validation
├── PredictionRun.py           # Inference script
├── ModelDeploy.py             # ONNX deployment
├── load_validation_results.py # Load validation results
├── 
├── # Loss functions
├── losses/
│   ├── __init__.py
│   ├── dice.py               # Dice loss
│   ├── focal.py              # Focal loss
│   ├── jaccard.py            # Jaccard loss
│   ├── lovasz.py             # Lovász loss
│   └── ...
├── 
├── # Utilities
├── Utils/
│   ├── __init__.py
│   ├── utils.py              # General utilities
│   ├── DataAugmentation.py   # Data augmentation
│   ├── network_modules.py    # Network building blocks
│   ├── deploy_onnxmodel.py   # ONNX deployment utilities
│   └── visualizeModel.py     # Model visualization
├── 
├── # Data and outputs
├── data/                      # Dataset directory
├── logs/                      # Training logs and checkpoints
└── deployed_model/            # Deployed models
```

## 📚 API Documentation

### Core Classes

- **`DataModel`**: Lightning DataModule for data loading and preprocessing
- **`NetModule`**: Lightning Module containing the model architecture and training logic
- **`CNNNet`**: Custom CNN architecture based on ResNet blocks with U-Net structure
- **`myDataset_img`**: Custom dataset class for image segmentation

### Key Functions

- **`listFiles()`**: Utility for file listing with natural sorting
- **`split_list()`**: Data splitting for train/test/validation
- **`k_fold_split()`**: K-fold cross validation splitting
- **`save_onnxmodel()`**: Model export to ONNX format
- **`find_best_checkpoint()`**: Automatic best checkpoint selection

## 📖 Complete Documentation

This README provides a quick overview. For comprehensive documentation:

- **[📚 Documentation Index](docs/README.md)** - Choose your learning path
- **[👤 User Guide](docs/user_guide.md)** - Step-by-step instructions and examples
- **[🔧 API Reference](docs/api.md)** - Complete technical documentation
- **[👨‍💻 Developer Guide](docs/developer_guide.md)** - Architecture and contribution guide

For detailed API documentation, see the [API Reference](docs/api.md).

## 💡 Examples

### Example 1: Medical Image Segmentation

```python
# Configure for medical images
[DataModule]
image_path = "./medical_data/images"
mask_path = "./medical_data/masks"
n_class = 4  # Background, organ1, organ2, organ3
image_shape = [512, 512, 1]  # Grayscale medical images
batch_size = 8  # Smaller batch for large images

[NetModule]
model_name = "medical_segmentation"
lr = 0.0001  # Lower learning rate for medical data
```

### Example 2: Satellite Image Segmentation

```python
# Configure for satellite imagery
[DataModule]
image_path = "./satellite_data/images"
mask_path = "./satellite_data/masks"
n_class = 6  # Different land use classes
image_shape = [256, 256, 3]  # RGB satellite images
batch_size = 32

[NetModule]
model_name = "satellite_segmentation"
lr = 0.001
```

### Example 3: Custom Loss Function

```python
# In NetModule.py, modify the training_step
def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.forward(x)
    y_hat_s = F.softmax(y_hat, dim=1)
    
    # Custom loss combination
    ce_loss = F.cross_entropy(y_hat, y)
    dice_loss = dice.DiceLoss(mode='multiclass')(y_hat_s, y)
    focal_loss = focal.FocalLoss()(y_hat, y)
    
    # Weighted combination
    total_loss = 0.5 * ce_loss + 0.3 * dice_loss + 0.2 * focal_loss
    
    self.log("train_loss", total_loss, on_epoch=True, prog_bar=True)
    return {'loss': total_loss}
```

## 🧪 Testing

### Unit Tests

Run the test suite:
```bash
python -m pytest tests/
```

### Model Architecture Test

Test the network architecture:
```bash
python Network.py
```

### Data Pipeline Test

Test data loading:
```bash
python DataPreprocessing.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/yukun-guo/PyTorchLightning-Template.git
cd PyTorchLightning-Template
pip install -e ".[dev]"
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch Lightning team for the excellent framework
- Contributors to the segmentation models library
- Open source community for various utilities and loss functions

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yukun-guo/PyTorchLightning-Template/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yukun-guo/PyTorchLightning-Template/discussions)
- **Documentation**: [Wiki](https://github.com/yukun-guo/PyTorchLightning-Template/wiki)

---

**Happy coding! 🚀**
