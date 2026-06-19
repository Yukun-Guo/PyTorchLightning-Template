"""
Shared training utilities.

This module centralizes the "best-practice" Lightning plumbing (logger,
callbacks, Trainer, checkpoint discovery) so the training / validation /
deployment scripts stay small and consistent. You normally do not need to
edit this file to adapt the template — change ``config.toml`` instead.
"""

from pathlib import Path
from typing import Optional

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger

try:
    import tomllib  # Python 3.11+

    def _load_toml(path: str) -> dict:
        with open(path, "rb") as f:
            return tomllib.load(f)
except ModuleNotFoundError:  # Python 3.10 fallback
    import toml

    def _load_toml(path: str) -> dict:
        return toml.load(path)


def load_config(config_path: str = "config.toml") -> dict:
    """Load the TOML configuration file."""
    if not Path(config_path).exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Run the scripts from the project root."
        )
    return _load_toml(config_path)


def checkpoint_dir(config: dict) -> Path:
    """Directory where checkpoints for this run are stored."""
    return Path(config["Logging"]["log_dir"]) / config["Project"]["name"] / "checkpoints"


def build_logger(config: dict) -> TensorBoardLogger:
    """TensorBoard logger writing to ``<log_dir>/<project name>``."""
    return TensorBoardLogger(
        save_dir=config["Logging"]["log_dir"],
        name=config["Project"]["name"],
        default_hp_metric=False,
    )


def build_callbacks(config: dict, fold: Optional[int] = None) -> list:
    """Build the standard callback stack: checkpointing, LR monitor, early stop."""
    train = config["Training"]
    logging = config["Logging"]
    name = config["Project"]["name"]
    monitor = train["monitor"]
    mode = train["monitor_mode"]

    fold_tag = f"fold{fold}-" if fold is not None else ""
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir(config)),
            filename=f"{name}-{fold_tag}{{epoch:03d}}-{{{monitor}:.4f}}",
            monitor=monitor,
            mode=mode,
            save_top_k=logging.get("save_top_k", 1),
            save_last=logging.get("save_last", True),
            auto_insert_metric_name=True,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if train.get("early_stopping", True):
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=train.get("early_stopping_patience", 15),
                min_delta=train.get("early_stopping_min_delta", 0.0),
                verbose=True,
            )
        )
    return callbacks


def build_trainer(config: dict, fold: Optional[int] = None, **overrides) -> L.Trainer:
    """Construct a Lightning ``Trainer`` from the ``[Training]`` config section."""
    train = config["Training"]
    clip = train.get("gradient_clip_val", 0.0)

    trainer_kwargs = dict(
        logger=build_logger(config),
        callbacks=build_callbacks(config, fold=fold),
        accelerator=train.get("accelerator", "auto"),
        devices=train.get("devices", "auto"),
        precision=train.get("precision", "32-true"),
        max_epochs=train.get("max_epochs", 100),
        gradient_clip_val=clip if clip else None,
        accumulate_grad_batches=train.get("accumulate_grad_batches", 1),
        log_every_n_steps=train.get("log_every_n_steps", 10),
        deterministic=train.get("deterministic", False),
    )
    trainer_kwargs.update(overrides)
    return L.Trainer(**trainer_kwargs)


def find_best_checkpoint(config: dict) -> Optional[str]:
    """Return the best checkpoint path for this run.

    Picks the checkpoint with the best monitored metric encoded in the filename
    (e.g. ``...-val_loss=0.1234.ckpt``), honouring ``monitor_mode``. Falls back
    to the most recently modified checkpoint when the metric cannot be parsed.
    """
    ckpt_dir = checkpoint_dir(config)
    if not ckpt_dir.exists():
        return None

    ckpts = [p for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt"]
    if not ckpts:
        ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return None

    monitor = config["Training"]["monitor"]
    mode = config["Training"]["monitor_mode"]
    best, best_score = None, None
    for ckpt in ckpts:
        token = f"{monitor}="
        if token in ckpt.name:
            try:
                score = float(ckpt.name.split(token)[1].split(".ckpt")[0])
            except (IndexError, ValueError):
                continue
            if best_score is None or (score < best_score if mode == "min" else score > best_score):
                best, best_score = ckpt, score

    if best is None:
        best = max(ckpts, key=lambda p: p.stat().st_mtime)
    return str(best)


def last_checkpoint(config: dict) -> Optional[str]:
    """Return the ``last.ckpt`` path if it exists (used to resume training)."""
    last = checkpoint_dir(config) / "last.ckpt"
    return str(last) if last.exists() else None
