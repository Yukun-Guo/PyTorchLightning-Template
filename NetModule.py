"""
PyTorch Lightning module for image segmentation.

Everything that defines *how the model learns* lives here:
  - the network architecture        -> :meth:`NetModule.build_model`
  - the loss function               -> :meth:`NetModule.compute_loss`
  - the optimizer + LR scheduler    -> :meth:`NetModule.configure_optimizers`
  - the train / val / test metrics  -> :meth:`NetModule._shared_step`

All hyper-parameters are read from ``config.toml`` so, in most cases, adapting
the template to your problem only requires editing that file. To use a custom
network, replace the body of :meth:`build_model`.
"""

from typing import Any, Dict, Tuple

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import functional as FM

import segmentation_models_pytorch as smp
from losses import dice


class NetModule(L.LightningModule):
    """LightningModule wrapping a segmentation network defined by ``config``."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # Persist the full config so the model can be re-created from a checkpoint
        # with ``NetModule.load_from_checkpoint(path)`` (no need to pass config again).
        self.save_hyperparameters()
        self.config = config

        data_cfg = config["DataModule"]
        # image_shape is stored as (C, H, W)
        self.in_channels = data_cfg["image_shape"][0]
        self.input_size = tuple(data_cfg["image_shape"][1:])
        self.n_class = data_cfg["n_class"]
        self.example_input_array = torch.randn(1, self.in_channels, *self.input_size)

        loss_cfg = config.get("Loss", {})
        self.ce_weight = float(loss_cfg.get("ce_weight", 1.0))
        self.dice_weight = float(loss_cfg.get("dice_weight", 1.0))

        self.model = self.build_model(config)

    # ------------------------------------------------------------------ #
    # >>> EDIT HERE to change the network architecture. <<<
    # Return any ``nn.Module`` mapping (B, C, H, W) -> (B, n_class, H, W).
    # ------------------------------------------------------------------ #
    def build_model(self, config: Dict[str, Any]) -> torch.nn.Module:
        model_cfg = config["Model"]
        encoder_weights = model_cfg.get("encoder_weights", "") or None
        return smp.create_model(
            arch=model_cfg.get("architecture", "Unet"),
            encoder_name=model_cfg.get("encoder_name", "resnet34"),
            encoder_weights=encoder_weights,
            in_channels=self.in_channels,
            classes=self.n_class,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ------------------------------------------------------------------ #
    # >>> EDIT HERE to change the loss function. <<<
    # ``logits`` are raw network outputs (B, n_class, H, W); ``y`` is (B, H, W).
    # ------------------------------------------------------------------ #
    def compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce_loss = F.cross_entropy(logits, y)
        dice_loss = dice.DiceLoss(mode="multiclass", from_logits=True)(logits, y)
        total = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return total, ce_loss, dice_loss

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss, ce_loss, dice_loss = self.compute_loss(logits, y)

        probs = logits.softmax(dim=1)
        iou = FM.jaccard_index(probs, y, task="multiclass", num_classes=self.n_class)
        acc = FM.accuracy(probs, y, task="multiclass", num_classes=self.n_class)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(
            {
                f"{stage}_ce_loss": ce_loss,
                f"{stage}_dice_loss": dice_loss,
                f"{stage}_iou": iou,
                f"{stage}_acc": acc,
            },
            prog_bar=(stage == "val"),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0) -> torch.Tensor:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return self(x).softmax(dim=1)

    def configure_optimizers(self):
        opt_cfg = self.config["Optimizer"]
        name = opt_cfg.get("name", "AdamW").lower()
        lr = float(opt_cfg.get("lr", 1e-3))
        weight_decay = float(opt_cfg.get("weight_decay", 0.0))

        if name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=lr, weight_decay=weight_decay,
                momentum=float(opt_cfg.get("momentum", 0.9)),
            )
        elif name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:  # default: AdamW (decoupled weight decay, a strong modern default)
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_name = opt_cfg.get("scheduler", "none").lower()
        monitor = self.config["Training"].get("monitor", "val_loss")

        if scheduler_name == "reducelronplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.config["Training"].get("monitor_mode", "min"),
                factor=float(opt_cfg.get("lr_factor", 0.5)),
                patience=int(opt_cfg.get("lr_patience", 5)),
                min_lr=float(opt_cfg.get("min_lr", 0.0)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": monitor, "interval": "epoch"},
            }
        if scheduler_name == "cosineannealinglr":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.config["Training"].get("max_epochs", 100)),
                eta_min=float(opt_cfg.get("min_lr", 0.0)),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        return optimizer

    def summary(self) -> None:
        """Print a layer-by-layer summary of the network."""
        try:
            from torchinfo import summary as _summary

            _summary(self.model, input_size=tuple(self.example_input_array.shape))
        except ImportError:
            print(self.model)


if __name__ == "__main__":
    from Utils.training import load_config

    model = NetModule(load_config("config.toml"))
    model.summary()
