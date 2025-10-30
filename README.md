# PyTorchLightning-Template

## 1. Create new env

```{bash}
python -m venv env
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

### or create new env using conda

```{bash}
conda create -n env python=3.x
conda install pip
```

## 2. Install pytorch

### Linux/windows

```{bash}
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### MacOS

```{bash}
pip3 install torch torchvision
```

## 3. Install requirements.txt

```{bash}
pip install -r requirements.txt
```

## 4. Install segmentation-models-pytorch

```{bash}
pip install git+https://github.com/qubvel-org/segmentation_models.pytorch
```
