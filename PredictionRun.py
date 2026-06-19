"""
Run inference on a folder of images and save colored segmentation masks.

    python PredictionRun.py

Loads the best checkpoint (or one set in ``[Deployment].checkpoint``), predicts
a class map for every image in ``[DataModule].image_path``, and writes a colored
PNG per image to ``<log_dir>/predicted_out``.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from NetModule import NetModule
from Utils.training import find_best_checkpoint, load_config
from Utils.utils import apply_colormap, listFiles


def preprocess_image(image_path: str, channels: int, size) -> torch.Tensor:
    """Load and preprocess one image to a (1, C, H, W) float tensor in [0, 1]."""
    mode = "L" if channels == 1 else "RGB"
    img = Image.open(image_path).convert(mode).resize((size[1], size[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if channels == 1:
        arr = arr[None, ...]          # (H, W) -> (1, H, W)
    else:
        arr = np.transpose(arr, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    return torch.from_numpy(arr).unsqueeze(0)


def main(config_path: str = "config.toml"):
    config = load_config(config_path)
    c, h, w = config["DataModule"]["image_shape"]

    checkpoint = config["Deployment"].get("checkpoint") or find_best_checkpoint(config)
    if not checkpoint:
        print("No checkpoint found. Train a model first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NetModule.load_from_checkpoint(checkpoint).eval().to(device)
    print(f"Loaded {checkpoint} on {device}")

    image_files = listFiles(config["DataModule"]["image_path"], "*.png")
    out_dir = Path(config["Logging"]["log_dir"]) / "predicted_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(image_files, start=1):
        x = preprocess_image(image_path, channels=c, size=(h, w)).to(device)
        with torch.no_grad():
            pred = model(x).softmax(dim=1).argmax(dim=1)
        class_mask = pred.squeeze().cpu().numpy().astype(np.uint8)
        colored = apply_colormap(class_mask).astype(np.uint8)
        Image.fromarray(colored).save(out_dir / f"{Path(image_path).stem}_pred.png")
        print(f"[{i}/{len(image_files)}] {os.path.basename(image_path)}")

    print(f"\nPredictions saved to: {out_dir}")


if __name__ == "__main__":
    main()
