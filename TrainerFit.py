"""
Standard training entry point.

    python TrainerFit.py

Reads ``config.toml``, builds the data module, model, and a Lightning Trainer
configured with best-practice callbacks (checkpointing, early stopping, LR
monitoring), then trains and runs a final test pass on the held-out set.
"""

import lightning as L

from DataModule import DataModel
from NetModule import NetModule
from Utils.training import build_trainer, last_checkpoint, load_config


def main(config_path: str = "config.toml"):
    config = load_config(config_path)
    L.seed_everything(config["Project"]["seed"], workers=True)

    data_module = DataModel(config=config)
    model = NetModule(config=config)

    trainer = build_trainer(config)

    # Optionally resume from the most recent checkpoint.
    ckpt_path = last_checkpoint(config) if config["Model"].get("restore_model", False) else None
    if ckpt_path:
        print(f"Resuming from checkpoint: {ckpt_path}")

    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

    # Evaluate the best checkpoint on the test split.
    trainer.test(model, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()
