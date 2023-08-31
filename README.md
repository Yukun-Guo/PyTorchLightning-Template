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

### windows

```{bash}
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### linux or mac

```{bash}
pip3 install torch torchvision torchaudio
```

## 3. Install requirements.txt

```{bash}
pip install -r requirements.txt
```

## 4. Install segmentation-models-pytorch

```{bash}
pip install git+https://github.com/qubvel/segmentation_models.pytorch
# or
pip install git+https://github.com/Yukun-Guo/segmentation_models_pytorch
```
