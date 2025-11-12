"""
PyTorch Lightning Module for image segmentation models.

This module defines the neural network architecture, training logic, optimization,
and callbacks for image segmentation tasks using PyTorch Lightning framework.
"""

import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics import functional as FM
from lightning.pytorch.callbacks import early_stopping, model_checkpoint, lr_monitor
from lightning.pytorch.loggers import TensorBoardLogger
from Network import CNNNet
from losses import dice
from torchsummary import summary
from typing import Dict, List, Any, Tuple


class NetModule(L.LightningModule):
    """
    Lightning Module for image segmentation training and inference.
    
    This class encapsulates the complete model definition including:
    - Neural network architecture (CNNNet)
    - Training and validation logic
    - Loss function computation (CrossEntropy + Dice)
    - Optimization configuration (Adam + ReduceLROnPlateau)
    - Callbacks setup (EarlyStopping, ModelCheckpoint, LRMonitor)
    - Logging configuration (TensorBoard)
    
    Args:
        config (dict): Configuration dictionary containing model settings.
            Required keys:
            - DataModule.image_shape: Input image dimensions [H, W, C]
            - DataModule.n_class: Number of segmentation classes
            - NetModule.model_name: Model name for logging and checkpoints
            - NetModule.log_dir: Directory for saving logs
            - DataModule.k_fold: Number of folds for cross-validation
    
    Attributes:
        input_size (tuple): Input image dimensions (height, width)
        img_chn (int): Number of input channels
        n_class (int): Number of output classes
        example_input_array (torch.Tensor): Example input for model summary
        out (CNNNet): The main network architecture
        model_name (str): Model name for identification
        log_dir (str): Directory for logging
        k_fold (int): Number of folds for cross-validation
    
    Example:
        >>> config = {
        ...     'DataModule': {
        ...         'image_shape': [256, 256, 1],
        ...         'n_class': 4,
        ...         'k_fold': 5
        ...     },
        ...     'NetModule': {
        ...         'model_name': 'segmentation_model',
        ...         'log_dir': './logs/'
        ...     }
        ... }
        >>> model = NetModule(config)
        >>> trainer = L.Trainer(max_epochs=100)
        >>> trainer.fit(model, datamodule=data_module)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the NetModule with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with model settings
        """
        super(NetModule, self).__init__()

        self.save_hyperparameters()
        
        self.input_size = config['DataModule']['image_shape'][:2]
        self.img_chn = config['DataModule']['image_shape'][2]
        self.n_class = config['DataModule']['n_class']
        self.example_input_array = torch.randn((1, self.img_chn, *self.input_size))
        self.out = CNNNet(
            in_channels=self.img_chn,
            out_channels=self.n_class,
            out_activation=None
        )

        self.model_name = config['NetModule']["model_name"]
        self.log_dir = config['NetModule']["log_dir"]
        self.k_fold = config['DataModule']["k_fold"]
        self.valid_dataset = None
        self.train_dataset = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, n_classes, height, width)
        """
        return self.out(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step for one batch.
        
        Computes the forward pass, calculates loss (CrossEntropy + Dice), and logs metrics.
        
        Args:
            batch (tuple): Batch containing (images, masks) tensors
            batch_idx (int): Index of the current batch
        
        Returns:
            dict: Dictionary containing the computed loss
        """
        x, y = batch
        y_hat = self.forward(x)
        y_hat_s = F.softmax(y_hat, dim=1, _stacklevel=5)
        
        # Combined loss: CrossEntropy + Dice
        ce_loss = F.cross_entropy(y_hat, y)
        dice_loss = dice.DiceLoss(mode='multiclass')(y_hat_s, y)
        train_loss = ce_loss + dice_loss
        
        self.log("train_loss", train_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_ce_loss", ce_loss, on_epoch=True, logger=True)
        self.log("train_dice_loss", dice_loss, on_epoch=True, logger=True)

        return {'loss': train_loss}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Validation step for one batch.
        
        Computes forward pass, calculates validation metrics (loss and IoU),
        and logs them for monitoring.
        
        Args:
            batch (tuple): Batch containing (images, masks) tensors
            batch_idx (int): Index of the current batch
        """
        x, y = batch
        y_hat = self.forward(x)
        y_hat_s = F.softmax(y_hat, dim=1, _stacklevel=5)
        
        # Validation loss
        ce_loss = F.cross_entropy(y_hat_s, y)
        dice_loss = dice.DiceLoss(mode='multiclass')(y_hat_s, y)
        val_loss = ce_loss + dice_loss
        
        # Validation IoU
        val_iou = FM.jaccard_index(y_hat_s, y, task='multiclass', num_classes=self.n_class)
        
        self.log_dict({
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_ce_loss': ce_loss,
            'val_dice_loss': dice_loss
        }, prog_bar=True, logger=True)


    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Any]]]:
        """
        Configure optimizers and learning rate schedulers.
        
        Sets up Adam optimizer with ReduceLROnPlateau scheduler that reduces
        learning rate when validation loss plateaus.
        
        Returns:
            tuple: (optimizers, schedulers) - Lists containing optimizer and scheduler configs
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002)
        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            min_lr=1e-8
        )
        lr_scheduler = {
            'scheduler': reduce_lr_on_plateau,
            'monitor': 'val_loss',
            'interval': 'epoch',
            'reduce_on_plateau': True
        }
        return [optimizer], [lr_scheduler]

    def configure_callbacks(self) -> List[L.Callback]:
        """
        Configure training callbacks.
        
        Sets up:
        - EarlyStopping: Stops training when validation loss stops improving
        - ModelCheckpoint: Saves best model based on validation loss
        - LearningRateMonitor: Logs learning rate changes
        
        Returns:
            list: List of configured callback instances
        """
        fd = str(self.k_fold)
        early_stop = early_stopping.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-08,
            patience=10,
            verbose=True
        )

        checkpoint = model_checkpoint.ModelCheckpoint(
            dirpath=self.log_dir + self.model_name,
            monitor="val_loss",
            save_top_k=1,
            verbose=True,
            filename=f'{self.model_name}-fold={fd}-{{epoch:03d}}-{{val_loss:.5f}}'
        )

        lr_monitors = lr_monitor.LearningRateMonitor(logging_interval='epoch')
        return [early_stop, checkpoint, lr_monitors]

    def configure_loggers(self) -> TensorBoardLogger:
        """
        Configure training loggers.
        
        Returns:
            TensorBoardLogger: TensorBoard logger for monitoring training progress
        """
        return TensorBoardLogger(self.log_dir, name=self.model_name)

    def summary(self) -> None:
        """
        Print model architecture summary using torchsummary.
        
        Automatically detects available device (CUDA or CPU) and prints
        detailed information about model layers, parameters, and memory usage.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        summary(self.to(device), tuple(self.example_input_array.shape[1:]))


if __name__ == '__main__':

    model = NetModule(CNNNet)
    model.summary()
    model.to_onnx('test.onnx')
