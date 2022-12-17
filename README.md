# PyTorchLightning-Template

## 1. create new env using python

```{bash}
python -m venv env
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

## or create new env using conda

```{bash}
conda create -n myenv python=3.x
conda install pip
```

## 2. install requirements.txt

```{bash}
pip install -r requirements.txt
```

## 3. install [pytorch](https://pytorch.org/get-started/locally/)

```{bash}
# windows

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# linux
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

## 4. install segmentation-models-pytorch

```{bash}
pip install git+https://github.com/qubvel/segmentation_models.pytorch
// or
pip install git+https://github.com/Yukun-Guo/segmentation_models_pytorch
```
