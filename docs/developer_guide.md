# Developer Guide

This guide is for developers who want to extend, modify, or contribute to the PyTorch Lightning Template. It covers the architecture, design patterns, and development workflows.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Development Setup](#development-setup)
3. [Code Structure](#code-structure)
4. [Design Patterns](#design-patterns)
5. [Extending the Template](#extending-the-template)
6. [Adding New Features](#adding-new-features)
7. [Testing](#testing)
8. [Performance Optimization](#performance-optimization)
9. [Contributing Guidelines](#contributing-guidelines)
10. [Release Process](#release-process)

## Architecture Overview

### High-Level Architecture

The template follows a modular architecture based on PyTorch Lightning's design principles:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   DataModule    │    │    NetModule     │    │     Trainer     │
│                 │    │                  │    │                 │
│ • Data loading  │    │ • Model def.     │    │ • Training loop │
│ • Preprocessing │◄──►│ • Loss/metrics   │◄──►│ • Validation    │
│ • Augmentation  │    │ • Optimization   │    │ • Checkpointing │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│     Dataset     │    │     Network      │    │   Callbacks     │
│                 │    │                  │    │                 │
│ • Data I/O      │    │ • Architecture   │    │ • Early stop    │
│ • Transforms    │    │ • Forward pass   │    │ • Checkpoints   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

1. **DataModule**: Handles all data-related operations
2. **NetModule**: Contains model definition and training logic
3. **Network**: Implements the neural network architecture
4. **Dataset**: Manages individual sample loading and transformation
5. **Utilities**: Helper functions and modules
6. **Loss Functions**: Specialized loss functions for segmentation

### Design Philosophy

- **Modularity**: Each component has a single responsibility
- **Configurability**: All parameters controlled via configuration files
- **Extensibility**: Easy to add new features without breaking existing code
- **Reproducibility**: Deterministic behavior with proper seeding
- **Scalability**: Works from single GPU to multi-node setups

## Development Setup

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd PyTorchLightning-Template
   ```

2. **Install uv (if not already installed):**
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Install dependencies with uv:**
   ```bash
   # Create virtual environment and install all dependencies including dev dependencies
   uv sync --all-extras
   
   # Activate the virtual environment
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### IDE Configuration

#### VS Code Settings
```json
{
    "python.defaultInterpreterPath": "./dev-env/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true
}
```

#### PyCharm Settings
- Enable Black formatter
- Configure isort with Black profile
- Set up flake8 as external tool
- Enable type checking with mypy

## Code Structure

### Module Responsibilities

#### DataModule.py
```python
class DataModel(L.LightningDataModule):
    """
    Responsibilities:
    - Data loading and splitting
    - Dataset creation
    - DataLoader configuration
    - Cross-validation setup
    """
```

#### NetModule.py
```python
class NetModule(L.LightningModule):
    """
    Responsibilities:
    - Model instantiation
    - Forward pass definition
    - Loss computation
    - Optimization configuration
    - Metric logging
    - Callback setup
    """
```

#### Network.py
```python
class CNNNet(nn.Module):
    """
    Responsibilities:
    - Network architecture definition
    - Layer composition
    - Forward pass implementation
    - Architecture-specific logic
    """
```

#### DataPreprocessing.py
```python
class myDataset_img(Dataset):
    """
    Responsibilities:
    - Individual sample loading
    - Data transformation pipeline
    - Augmentation application
    - Format conversion
    """
```

### Configuration Management

The template uses TOML for configuration management:

```python
# Loading configuration
import toml
config = toml.load("config.toml")

# Accessing nested values
batch_size = config['DataModule']['batch_size']
lr = config['NetModule']['lr']

# Configuration validation
def validate_config(config):
    required_keys = ['DataModule', 'NetModule']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing configuration section: {key}")
```

### Error Handling Patterns

```python
# Custom exceptions
class TemplateError(Exception):
    """Base exception for template-specific errors."""
    pass

class DataError(TemplateError):
    """Raised when data-related issues occur."""
    pass

class ModelError(TemplateError):
    """Raised when model-related issues occur."""
    pass

# Error handling in methods
def load_data(self):
    try:
        data = self._load_raw_data()
    except FileNotFoundError as e:
        raise DataError(f"Data files not found: {e}")
    except Exception as e:
        raise DataError(f"Unexpected error loading data: {e}")
```

## Design Patterns

### Factory Pattern for Model Creation

```python
class ModelFactory:
    """Factory for creating different model architectures."""
    
    @staticmethod
    def create_model(model_type: str, config: dict):
        if model_type == "cnn":
            return CNNNet(**config)
        elif model_type == "unet":
            return UNet(**config)
        elif model_type == "deeplabv3":
            return DeepLabV3(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
```

### Strategy Pattern for Loss Functions

```python
class LossStrategy:
    """Base class for loss function strategies."""
    
    def compute_loss(self, predictions, targets):
        raise NotImplementedError

class DiceLossStrategy(LossStrategy):
    def __init__(self, smooth=1e-6):
        self.smooth = smooth
    
    def compute_loss(self, predictions, targets):
        return dice_loss(predictions, targets, self.smooth)

class FocalLossStrategy(LossStrategy):
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
    
    def compute_loss(self, predictions, targets):
        return focal_loss(predictions, targets, self.alpha, self.gamma)
```

### Observer Pattern for Metrics

```python
class MetricObserver:
    """Observer for metric computation and logging."""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, name: str, value: float, step: int):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((step, value))
    
    def get_history(self, name: str):
        return self.metrics.get(name, [])
```

### Builder Pattern for Complex Configurations

```python
class ConfigBuilder:
    """Builder for creating complex configurations."""
    
    def __init__(self):
        self.config = {}
    
    def set_data_config(self, image_path, mask_path, n_class):
        self.config['DataModule'] = {
            'image_path': image_path,
            'mask_path': mask_path,
            'n_class': n_class
        }
        return self
    
    def set_model_config(self, model_name, lr, epochs):
        self.config['NetModule'] = {
            'model_name': model_name,
            'lr': lr,
            'epochs': epochs
        }
        return self
    
    def build(self):
        return self.config.copy()

# Usage
config = (ConfigBuilder()
          .set_data_config("./data/images", "./data/masks", 4)
          .set_model_config("my_model", 0.001, 100)
          .build())
```

## Extending the Template

### Adding New Loss Functions

1. **Create the loss function:**
   ```python
   # losses/new_loss.py
   import torch
   import torch.nn as nn
   from ._functional import soft_dice_score
   
   class NewLoss(nn.Module):
       def __init__(self, smooth=1e-6):
           super().__init__()
           self.smooth = smooth
       
       def forward(self, y_pred, y_true):
           # Implement your loss logic here
           return loss_value
   ```

2. **Register in losses/__init__.py:**
   ```python
   from .new_loss import NewLoss
   ```

3. **Use in NetModule:**
   ```python
   from losses import NewLoss
   
   def training_step(self, batch, batch_idx):
       x, y = batch
       y_hat = self.forward(x)
       loss = NewLoss()(y_hat, y)
       return {'loss': loss}
   ```

### Adding New Network Architectures

1. **Create the network file:**
   ```python
   # networks/new_network.py
   import torch.nn as nn
   from Utils.network_modules import Conv2dReLU, Activation
   
   class NewNetwork(nn.Module):
       def __init__(self, in_channels, out_channels, **kwargs):
           super().__init__()
           # Define your architecture here
       
       def forward(self, x):
           # Implement forward pass
           return x
   ```

2. **Register in Network.py or create a factory:**
   ```python
   from networks.new_network import NewNetwork
   
   def create_network(arch_type, **kwargs):
       if arch_type == "cnn":
           return CNNNet(**kwargs)
       elif arch_type == "new":
           return NewNetwork(**kwargs)
   ```

### Adding New Data Augmentations

1. **Create augmentation class:**
   ```python
   # Utils/DataAugmentation.py
   class NewAugmentation:
       def __init__(self, parameter=0.5):
           self.parameter = parameter
       
       def __call__(self, sample):
           img, mask = sample['img'], sample['mask']
           # Apply your augmentation
           return {'img': augmented_img, 'mask': augmented_mask}
   ```

2. **Add to transform pipeline:**
   ```python
   # DataPreprocessing.py
   self.transform = transforms.Compose([
       GrayJitter(),
       NewAugmentation(parameter=0.3),
       RandomFlip(axis=1),
       RandomCrop2D(out_size),
       Normalize(),
       ToTensor()
   ])
   ```

### Creating Custom Callbacks

```python
# callbacks/custom_callback.py
import lightning as L

class CustomCallback(L.Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training started!")
    
    def on_epoch_end(self, trainer, pl_module):
        # Custom logic at epoch end
        if trainer.current_epoch % 10 == 0:
            self.custom_validation(trainer, pl_module)
    
    def custom_validation(self, trainer, pl_module):
        # Implement custom validation logic
        pass

# Register in NetModule.py
def configure_callbacks(self):
    callbacks = [
        early_stopping.EarlyStopping(...),
        model_checkpoint.ModelCheckpoint(...),
        CustomCallback()
    ]
    return callbacks
```

## Adding New Features

### Feature Development Workflow

1. **Plan the feature:**
   - Define requirements and scope
   - Design the API and integration points
   - Consider backward compatibility

2. **Create feature branch:**
   ```bash
   git checkout -b feature/new-feature-name
   ```

3. **Implement the feature:**
   - Write code following existing patterns
   - Add comprehensive tests
   - Update documentation

4. **Test thoroughly:**
   - Unit tests for individual components
   - Integration tests for feature workflow
   - End-to-end tests with real data

5. **Create pull request:**
   - Clear description of changes
   - Link to related issues
   - Request appropriate reviewers

### Example: Adding Multi-Scale Training

1. **Modify DataModule:**
   ```python
   class MultiScaleDataModel(DataModel):
       def __init__(self, config):
           super().__init__(config)
           self.scales = config['DataModule'].get('scales', [256, 384, 512])
           self.current_scale = self.scales[0]
       
       def update_scale(self, epoch):
           # Change scale based on epoch
           scale_idx = (epoch // 20) % len(self.scales)
           self.current_scale = self.scales[scale_idx]
   ```

2. **Modify Dataset:**
   ```python
   def __getitem__(self, item):
       # Use current scale for resizing
       transform = self.get_transform(self.data_module.current_scale)
       return transform(sample)
   ```

3. **Add callback for scale updates:**
   ```python
   class MultiScaleCallback(L.Callback):
       def on_epoch_start(self, trainer, pl_module):
           data_module = trainer.datamodule
           data_module.update_scale(trainer.current_epoch)
   ```

### Adding Visualization Features

```python
# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np

class SegmentationVisualizer:
    def __init__(self, class_names, colormap=None):
        self.class_names = class_names
        self.colormap = colormap or self._default_colormap()
    
    def visualize_prediction(self, image, mask, prediction, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Input Image')
        
        axes[1].imshow(mask, cmap='tab10')
        axes[1].set_title('Ground Truth')
        
        axes[2].imshow(prediction, cmap='tab10')
        axes[2].set_title('Prediction')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def _default_colormap(self):
        return plt.cm.tab10
```

## Testing

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_datamodule.py
│   ├── test_netmodule.py
│   ├── test_network.py
│   └── test_utils.py
├── integration/             # Integration tests
│   ├── test_training_pipeline.py
│   └── test_inference_pipeline.py
├── e2e/                     # End-to-end tests
│   └── test_full_workflow.py
├── fixtures/                # Test data and fixtures
│   ├── sample_data/
│   └── configs/
└── conftest.py             # Pytest configuration
```

### Unit Test Examples

```python
# tests/unit/test_datamodule.py
import pytest
import tempfile
import os
from pathlib import Path
from DataModule import DataModel

class TestDataModel:
    @pytest.fixture
    def sample_config(self):
        return {
            'DataModule': {
                'image_path': 'test_images',
                'mask_path': 'test_masks',
                'batch_size': 4,
                'image_shape': [256, 256, 1],
                'split_ratio': [0.7, 0.2, 0.1],
                'shuffle': True
            }
        }
    
    @pytest.fixture
    def temp_data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample data structure
            img_dir = Path(tmpdir) / 'test_images'
            mask_dir = Path(tmpdir) / 'test_masks'
            img_dir.mkdir()
            mask_dir.mkdir()
            
            # Create dummy files
            for i in range(10):
                (img_dir / f'image_{i:03d}.png').touch()
                (mask_dir / f'image_{i:03d}.png').touch()
            
            yield tmpdir
    
    def test_initialization(self, sample_config, temp_data_dir):
        sample_config['DataModule']['image_path'] = str(Path(temp_data_dir) / 'test_images')
        sample_config['DataModule']['mask_path'] = str(Path(temp_data_dir) / 'test_masks')
        
        data_module = DataModel(sample_config)
        assert data_module.batch_size == 4
        assert data_module.img_shape == [256, 256, 1]
    
    def test_setup(self, sample_config, temp_data_dir):
        sample_config['DataModule']['image_path'] = str(Path(temp_data_dir) / 'test_images')
        sample_config['DataModule']['mask_path'] = str(Path(temp_data_dir) / 'test_masks')
        
        data_module = DataModel(sample_config)
        data_module.setup()
        
        assert len(data_module.train_dataset) == 7  # 70% of 10
        assert len(data_module.test_dataset) == 2   # 20% of 10
        assert len(data_module.valid_dataset) == 1  # 10% of 10
```

### Integration Test Examples

```python
# tests/integration/test_training_pipeline.py
import pytest
import tempfile
from pathlib import Path
import lightning as L
from DataModule import DataModel
from NetModule import NetModule

class TestTrainingPipeline:
    @pytest.fixture
    def integration_config(self):
        return {
            'DataModule': {
                'image_path': 'test_images',
                'mask_path': 'test_masks',
                'n_class': 3,
                'image_shape': [64, 64, 1],  # Small for testing
                'batch_size': 2,
                'split_ratio': [0.8, 0.2, 0.0]
            },
            'NetModule': {
                'model_name': 'test_model',
                'epochs': 2,  # Few epochs for testing
                'lr': 0.01,
                'log_dir': './test_logs/',
                'k_fold': 1
            }
        }
    
    def test_complete_training_cycle(self, integration_config, sample_data):
        data_module = DataModel(integration_config)
        net_module = NetModule(integration_config)
        
        trainer = L.Trainer(
            max_epochs=2,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False
        )
        
        # Should not raise any exceptions
        trainer.fit(net_module, datamodule=data_module)
        
        # Verify model was trained
        assert trainer.current_epoch == 1  # 0-indexed
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_datamodule.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run only integration tests
pytest tests/integration/

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_data"
```

## Performance Optimization

### Profiling Code

```python
# profiling/profile_training.py
import cProfile
import pstats
from DataModule import DataModel
from NetModule import NetModule

def profile_training():
    config = load_config()
    data_module = DataModel(config)
    net_module = NetModule(config)
    
    # Profile data loading
    pr = cProfile.Profile()
    pr.enable()
    data_module.setup()
    pr.disable()
    
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

if __name__ == "__main__":
    profile_training()
```

### Memory Optimization

```python
# utils/memory_utils.py
import torch
import gc

class MemoryManager:
    @staticmethod
    def clear_cache():
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage."""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated(),
                'cached': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated()
            }
        return {'cpu_percent': psutil.virtual_memory().percent}
    
    @contextmanager
    def memory_profiler(self):
        """Context manager for memory profiling."""
        start_memory = self.get_memory_usage()
        try:
            yield
        finally:
            end_memory = self.get_memory_usage()
            print(f"Memory usage change: {end_memory['allocated'] - start_memory['allocated']} bytes")
```

### Data Loading Optimization

```python
# Optimized DataLoader configuration
def get_optimized_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=min(8, os.cpu_count()),  # Optimal worker count
        pin_memory=True,                     # For GPU training
        persistent_workers=True,             # Keep workers alive
        prefetch_factor=2,                   # Prefetch batches
        drop_last=True                       # For consistent batch sizes
    )
```

### Model Optimization

```python
# Model compilation (PyTorch 2.0+)
def get_compiled_model(model):
    if hasattr(torch, 'compile'):
        return torch.compile(model)
    return model

# Mixed precision training
trainer = L.Trainer(
    precision="16-mixed",  # Use automatic mixed precision
    max_epochs=100
)
```

## Contributing Guidelines

### Code Style

1. **Follow PEP 8**: Use Black formatter with line length 88
2. **Import sorting**: Use isort with Black profile
3. **Type hints**: Add type hints to all public functions
4. **Docstrings**: Use Google-style docstrings

```python
def example_function(param1: int, param2: str = "default") -> bool:
    """Brief description of the function.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
    
    Returns:
        Description of return value.
    
    Raises:
        ValueError: Description of when this error is raised.
    """
    return True
```

### Commit Messages

Follow conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(datamodule): add k-fold cross validation support

- Implement k-fold splitting in DataModel
- Add configuration options for k-fold parameters
- Update tests for new functionality

Closes #123
```

### Pull Request Process

1. **Create feature branch**: `git checkout -b feature/feature-name`
2. **Make changes**: Follow coding standards
3. **Add tests**: Ensure good test coverage
4. **Update docs**: Update relevant documentation
5. **Run checks**: Ensure all tests and linting pass
6. **Create PR**: Use the provided template
7. **Address feedback**: Respond to review comments
8. **Merge**: Once approved, merge using squash and merge

### Code Review Checklist

**Functionality:**
- [ ] Code works as intended
- [ ] Edge cases are handled
- [ ] Error handling is appropriate
- [ ] Performance impact is acceptable

**Code Quality:**
- [ ] Code follows style guidelines
- [ ] Code is readable and well-commented
- [ ] No code duplication
- [ ] Appropriate abstractions are used

**Testing:**
- [ ] Adequate test coverage
- [ ] Tests are meaningful and test the right things
- [ ] Tests pass consistently

**Documentation:**
- [ ] Code is self-documenting
- [ ] Public APIs are documented
- [ ] README is updated if needed
- [ ] Examples are provided where appropriate

## Release Process

### Version Management

The project follows semantic versioning (SemVer):
- `MAJOR.MINOR.PATCH`
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes (backward compatible)

### Release Workflow

1. **Prepare release branch:**
   ```bash
   git checkout -b release/v1.2.0
   ```

2. **Update version numbers:**
   - `pyproject.toml`
   - `config.toml`
   - `__init__.py` files

3. **Update changelog:**
   ```markdown
   ## [1.2.0] - 2024-01-15
   
   ### Added
   - New k-fold validation feature
   - Support for custom loss functions
   
   ### Changed
   - Improved memory usage in data loading
   
   ### Fixed
   - Fixed bug in checkpoint loading
   ```

4. **Run full test suite:**
   ```bash
   pytest tests/
   python -m flake8
   python -m mypy .
   ```

5. **Create release PR:**
   - Review all changes
   - Ensure documentation is updated
   - Get approval from maintainers

6. **Tag release:**
   ```bash
   git tag -a v1.2.0 -m "Release version 1.2.0"
   git push origin v1.2.0
   ```

7. **Create GitHub release:**
   - Use tag as release title
   - Copy changelog as release notes
   - Attach relevant assets

8. **Post-release tasks:**
   - Merge release branch to main
   - Update development branch
   - Announce release in appropriate channels

This completes the comprehensive developer documentation. The template now has detailed guides for developers who want to understand, extend, or contribute to the codebase.