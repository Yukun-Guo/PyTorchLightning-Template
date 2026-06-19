# PyTorch Lightning Segmentation Template

A clean, **config-driven** template for training image-segmentation models with
[PyTorch Lightning](https://lightning.ai/) and
[segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).
It ships with sensible defaults for the things that matter in practice —
**early stopping, learning-rate scheduling, checkpointing, logging, mixed
precision, gradient clipping, and ONNX deployment** — so you can focus on your
data and model.

Adapting it to a new problem usually means editing a single file:
[`config.toml`](config.toml).

---

## Project layout

```
config.toml            # ← single source of truth: edit this first
DataModule.py          # how data is discovered, split, and batched
DataPreprocessing.py   # Dataset + transforms (augmentation, resize, normalize)
NetModule.py           # the model: architecture, loss, metrics, optimizer
TrainerFit.py          # train (standard) entry point
TrainerFitKFold.py     # train (k-fold cross validation) entry point
PredictionVal.py       # evaluate a checkpoint, dump metrics + predictions
PredictionRun.py       # run inference on a folder of images -> colored masks
ModelDeploy.py         # export the best checkpoint to ONNX and verify it
Utils/
  training.py          # shared Trainer/callback/logger builders (best practices)
  DataAugmentation.py  # a library of augmentation transforms
  utils.py             # file listing, splitting, colormap helpers
losses/                # Dice / Focal / Jaccard / Lovasz / ... loss functions
data/                  # example images + masks
```

---

## Quick start

```bash
# 1. Install dependencies (uv recommended; pip also works)
uv sync           # or:  pip install -e .

# 2. Train
python TrainerFit.py

# 3. Watch training
tensorboard --logdir logs

# 4. Evaluate the best checkpoint
python PredictionVal.py

# 5. Run inference on a folder of images
python PredictionRun.py

# 6. Export to ONNX
python ModelDeploy.py
```

Checkpoints are written to `logs/<project name>/checkpoints/` and TensorBoard
logs to `logs/<project name>/`.

---

## Adapting the template to your data

### 1. Point it at your images and masks — `[DataModule]`

```toml
[DataModule]
image_path  = "./data/images"
mask_path   = "./data/masks"
image_shape = [1, 480, 288]   # (C, H, W): C=1 grayscale, C=3 RGB
n_class     = 12              # number of classes, including background
batch_size  = 16
split_ratio = [0.7, 0.15, 0.15]  # [train, val, test]
augmentation = true           # applied to the TRAINING split only
```

Requirements for the default loader:
- Images and masks are PNGs, paired by sorted filename order.
- Masks are **indexed** images where each pixel value is its class id (`0..n_class-1`).

Different on-disk format (e.g. `.npy` volumes, JSON annotations, separate
naming)? Edit `setup()` in [`DataModule.py`](DataModule.py) and the file reading
in [`DataPreprocessing.py`](DataPreprocessing.py).

### 2. Choose the augmentations — `DataPreprocessing.py`

Augmentations live in [`build_transforms`](DataPreprocessing.py). The pipeline is
applied to the training split only; validation/test use just resize + normalize.
Many ready-made transforms (gamma, blur, elastic, noise, rolling, ...) are
available in [`Utils/DataAugmentation.py`](Utils/DataAugmentation.py).

### 3. Pick a model — `[Model]`

```toml
[Model]
architecture    = "Unet"      # Unet, UnetPlusPlus, FPN, DeepLabV3Plus, ...
encoder_name    = "resnet34"  # resnet34, efficientnet-b0, mit_b0, ...
encoder_weights = "imagenet"  # "" = random init, "imagenet" = pretrained
```

These map directly to `smp.create_model(...)`. To use a fully custom network,
replace the body of [`build_model`](NetModule.py) — it just needs to return an
`nn.Module` that maps `(B, C, H, W) -> (B, n_class, H, W)`.

### 4. Tune the loss — `[Loss]`

```toml
[Loss]
ce_weight   = 1.0   # CrossEntropy weight
dice_weight = 1.0   # Dice weight
```

The total loss is `ce_weight * CE + dice_weight * Dice`. Edit
[`compute_loss`](NetModule.py) to use a different combination (focal, jaccard,
lovász — all in [`losses/`](losses)).

---

## Best-practice training settings

These are configured in `[Optimizer]`, `[Training]` and `[Logging]` and wired up
in [`Utils/training.py`](Utils/training.py):

| Setting | Where | Default | Notes |
|---|---|---|---|
| Optimizer | `[Optimizer].name` | `AdamW` | decoupled weight decay; strong modern default |
| LR scheduler | `[Optimizer].scheduler` | `ReduceLROnPlateau` | also supports `CosineAnnealingLR` / `none` |
| Early stopping | `[Training].early_stopping` | `true` | patience `15`, monitors `val_loss` |
| Checkpointing | `[Logging].save_top_k` | best `1` + `last` | best by `monitor`, resumable via `last.ckpt` |
| Mixed precision | `[Training].precision` | `32-true` | set `16-mixed` for a big speed-up on modern GPUs |
| Gradient clipping | `[Training].gradient_clip_val` | `1.0` | `0` to disable |
| Grad accumulation | `[Training].accumulate_grad_batches` | `1` | raise to simulate a larger batch size |
| LR monitoring | always on | — | learning rate logged to TensorBoard each epoch |
| Reproducibility | `[Project].seed` | `1234` | `[Training].deterministic = true` for exact runs |

The metric used for **early stopping** and **best-checkpoint selection** is set
once via `[Training].monitor` / `monitor_mode` (e.g. `val_loss`/`min` or
`val_iou`/`max`) and is reused everywhere.

### Resume training

Set `[Model].restore_model = true` and re-run `python TrainerFit.py`; it picks up
from `logs/<project name>/checkpoints/last.ckpt`.

---

## Deployment

`python ModelDeploy.py` exports the best checkpoint to ONNX (with Conv→BN
fusion), verifies the ONNX output against PyTorch, and writes a CoreML
compatibility report. Configure it in `[Deployment]`:

```toml
[Deployment]
checkpoint    = ""           # "" = auto-pick the best checkpoint
output_dir    = "./deployed_model"
onnx_filename = "model.onnx"
opset         = 18
quantize      = false
```

The exported model uses dynamic batch/height/width axes, so it accepts variable
input sizes at inference time.

---

## Tips

- On Windows, set `[DataModule].num_workers = 0` if you hit DataLoader worker errors.
- For class-imbalanced data, monitor `val_iou` (`monitor_mode = "max"`) instead of `val_loss`.
- Start with `encoder_weights = "imagenet"` for faster convergence on small datasets.
