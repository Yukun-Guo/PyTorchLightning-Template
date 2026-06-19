"""
Evaluate the best checkpoint and save per-sample predictions + metrics.

    python PredictionVal.py

Loads the best checkpoint, runs the validation split, stores the aggregated
metrics (JSON) and per-sample prediction / ground-truth arrays under
``<log_dir>/validation_results/<timestamp>``.
"""

import json
from datetime import datetime
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import functional as FM

from DataModule import DataModel
from NetModule import NetModule
from Utils.training import find_best_checkpoint, load_config


def main(config_path: str = "config.toml"):
    config = load_config(config_path)
    L.seed_everything(config["Project"]["seed"], workers=True)

    checkpoint = config["Deployment"].get("checkpoint") or find_best_checkpoint(config)
    if not checkpoint:
        print("No checkpoint found. Train a model first.")
        return
    print(f"Evaluating checkpoint: {checkpoint}")

    n_class = config["DataModule"]["n_class"]
    data_module = DataModel(config=config)
    data_module.setup()

    model = NetModule.load_from_checkpoint(checkpoint).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config["Logging"]["log_dir"]) / "validation_results" / timestamp
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    losses, ious, sample_idx = [], [], 0
    with torch.no_grad():
        for x, y in data_module.val_dataloader():
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = logits.softmax(dim=1)
            losses.append(F.cross_entropy(logits, y).item())
            ious.append(FM.jaccard_index(probs, y, task="multiclass", num_classes=n_class).item())

            preds = probs.argmax(dim=1).cpu().numpy().astype(np.uint8)
            targets = y.cpu().numpy().astype(np.uint8)
            for pred, target in zip(preds, targets):
                np.save(pred_dir / f"val_{sample_idx:04d}_prediction.npy", pred)
                np.save(pred_dir / f"val_{sample_idx:04d}_ground_truth.npy", target)
                sample_idx += 1

    metrics = {
        "validation_loss": float(np.mean(losses)),
        "validation_iou": float(np.mean(ious)),
        "num_samples": sample_idx,
        "timestamp": datetime.now().isoformat(),
        "checkpoint": checkpoint,
    }
    with open(out_dir / "validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Loss: {metrics['validation_loss']:.4f} | IoU: {metrics['validation_iou']:.4f} "
          f"| samples: {metrics['num_samples']}")
    print(f"Results saved to: {out_dir}")
    return metrics


if __name__ == "__main__":
    main()
