# Documentation Index

Welcome to the PyTorch Lightning Template documentation! This comprehensive template provides everything you need to build image segmentation models quickly and effectively.

## 📚 Documentation Overview

### Quick Start
- **[README.md](../README.md)** - Main overview, installation, and quick start guide
- **[User Guide](user_guide.md)** - Step-by-step instructions for using the template

### Detailed Documentation
- **[API Reference](api.md)** - Complete API documentation for all classes and functions
- **[Developer Guide](developer_guide.md)** - Architecture details and contribution guidelines

## 🎯 Choose Your Path

### I'm New to This Template
Start with the [README.md](../README.md) for installation and basic usage, then move to the [User Guide](user_guide.md) for detailed examples.

### I Want to Use the Template
The [User Guide](user_guide.md) provides comprehensive step-by-step instructions for:
- Data preparation
- Configuration
- Training workflows
- Validation and testing
- Model deployment

### I Need Technical Details
The [API Reference](api.md) contains detailed documentation for:
- All classes and their methods
- Function parameters and return types
- Usage examples
- Configuration schemas

### I Want to Contribute or Extend
The [Developer Guide](developer_guide.md) covers:
- Architecture overview
- Code structure and patterns
- Adding new features
- Testing guidelines
- Contribution workflow

## 🔍 Quick Reference

### Core Components
- **DataModule**: Handles data loading and preprocessing
- **NetModule**: Contains the Lightning model and training logic
- **CNNNet**: Implements the neural network architecture
- **Dataset Classes**: Manage individual sample loading and transforms

### Key Files
```
├── DataModule.py              # Lightning DataModule
├── NetModule.py               # Lightning Module (model + training)
├── Network.py                 # CNN architecture
├── DataPreprocessing.py       # Dataset and transforms
├── TrainerFit.py              # Training script
├── config.toml                # Configuration file
└── docs/                      # All documentation
    ├── api.md                 # API reference
    ├── user_guide.md          # User guide
    └── developer_guide.md     # Developer guide
```

### Configuration Quick Reference
```toml
[DataModule]
image_path = "./data/images"     # Input images directory
mask_path = "./data/masks"       # Ground truth masks directory
n_class = 4                      # Number of classes
image_shape = [256, 256, 1]      # [height, width, channels]
batch_size = 16                  # Training batch size

[NetModule]
model_name = "my_model"          # Model identifier
lr = 0.001                       # Learning rate
epochs = 200                     # Maximum epochs
```

## 💡 Common Use Cases

### Medical Image Segmentation
```toml
n_class = 3                      # Background + organs
image_shape = [512, 512, 1]      # High resolution grayscale
batch_size = 4                   # Large images need smaller batches
```

### Satellite Image Segmentation
```toml
n_class = 6                      # Land use classes
image_shape = [256, 256, 3]      # RGB satellite images
batch_size = 32                  # Smaller images allow larger batches
```

### Automotive Segmentation
```toml
n_class = 8                      # Road elements
image_shape = [384, 216, 3]      # 16:9 aspect ratio
```

## 🛠️ Getting Help

### Documentation Issues
If you find errors or unclear sections in the documentation:
1. Check the [API Reference](api.md) for technical details
2. Review [examples](user_guide.md#examples) in the User Guide
3. Open an issue on GitHub

### Usage Questions
For questions about using the template:
1. Check the [User Guide](user_guide.md) troubleshooting section
2. Review the [configuration examples](user_guide.md#configuration)
3. Look at similar use cases in the documentation

### Development Questions
For questions about extending or modifying the template:
1. Review the [Developer Guide](developer_guide.md)
2. Check existing code patterns
3. Look at the test examples

## 📈 Next Steps

1. **Install the template** following the [README](../README.md)
2. **Prepare your data** using the [User Guide](user_guide.md#data-preparation)
3. **Configure the model** with the [configuration guide](user_guide.md#configuration)
4. **Start training** using the [training instructions](user_guide.md#training)
5. **Deploy your model** following the [deployment guide](user_guide.md#model-deployment)

## 🎉 Happy Segmenting!

This template is designed to get you from data to deployed model as quickly as possible while maintaining flexibility for advanced use cases. Whether you're a researcher, student, or industry practitioner, you should find everything you need to build state-of-the-art segmentation models.

---

**Need more help?** Open an issue on GitHub or check our community discussions!