import os
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import lightning as L
import torch.nn.functional as F
from torchmetrics import functional as FM
import toml
from NetModule import NetModule
from DataModule import DataModel
from Utils.utils import listFiles, split_list

def find_best_checkpoint(checkpoint_dir, model_name):
    """Find the best checkpoint file based on validation loss."""
    checkpoint_path = Path(checkpoint_dir) / model_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    
    # Look for checkpoint files
    ckpt_files = list(checkpoint_path.glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_path}")
    
    # Find the best checkpoint (lowest validation loss)
    best_ckpt = None
    best_val_loss = float('inf')
    
    for ckpt_file in ckpt_files:
        filename = ckpt_file.name
        if 'val_loss' in filename:
            try:
                # Extract validation loss from filename
                val_loss_str = filename.split('val_loss=')[1].split('.ckpt')[0]
                val_loss = float(val_loss_str)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt = ckpt_file
            except (IndexError, ValueError):
                continue
    
    if best_ckpt is None:
        # If no validation loss in filename, use the most recent file
        best_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
    
    print(f"Using checkpoint: {best_ckpt}")
    return str(best_ckpt)

def save_validation_results(predictions, targets, metrics, save_dir, filenames=None):
    """Save validation predictions and metrics to files."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions as numpy arrays
    pred_dir = save_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        # Convert to numpy if tensor
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()
            
        # Save prediction and ground truth
        filename = f"{i:03d}" if filenames is None else filenames[i].stem
        np.save(pred_dir / f"{filename}_prediction.npy", pred)
        np.save(pred_dir / f"{filename}_ground_truth.npy", target)
    
    # Save metrics as JSON
    metrics_file = save_dir / "validation_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Validation results saved to: {save_dir}")

def run_validation():
    """Main function to run validation and save results."""
    # Set random seed for reproducibility
    L.seed_everything(1234)
    
    # Load configuration
    try:
        with open('config.toml', 'r') as f:
            config = toml.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("config.toml not found. Please ensure it exists in the project root.")
    
    print("Configuration loaded successfully.")
    
    # Initialize data module
    data_model = DataModel(config=config)
    data_model.setup()
    
    # Find and load the best checkpoint
    try:
        model_path = find_best_checkpoint(
            config['NetModule']['log_dir'], 
            config['NetModule']['model_name']
        )
        net_model = NetModule.load_from_checkpoint(model_path, config=config)
        print(f"Model loaded from: {model_path}")
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        return
    
    # Create trainer for validation
    trainer = L.Trainer(
        logger=False,  # Disable logging for validation
        devices=1 if torch.cuda.is_available() else 0,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        enable_progress_bar=True,
        enable_model_summary=False
    )
    
    # Run validation to get metrics
    print("Running validation...")
    val_results = trainer.validate(model=net_model, datamodule=data_model, verbose=True)
    
    # Collect predictions manually for saving
    net_model.eval()
    device = next(net_model.parameters()).device
    
    all_predictions = []
    all_targets = []
    all_losses = []
    all_ious = []
    filenames = []
    
    val_dataloader = data_model.val_dataloader()
    
    print("Collecting predictions...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            y_hat = net_model(x)
            y_hat_softmax = F.softmax(y_hat, dim=1)
            
            # Calculate metrics
            batch_loss = F.cross_entropy(y_hat, y)
            batch_iou = FM.jaccard_index(y_hat_softmax, y, task='multiclass', 
                                       num_classes=config['DataModule']['n_class'])
            
            # Store results
            all_predictions.extend(y_hat_softmax.cpu())
            all_targets.extend(y.cpu())
            all_losses.append(batch_loss.item())
            all_ious.append(batch_iou.item())
            
            # Generate filenames for this batch
            batch_size = x.size(0)
            start_idx = batch_idx * val_dataloader.batch_size
            batch_filenames = [f"val_{start_idx + i:03d}" for i in range(batch_size)]
            filenames.extend(batch_filenames)
    
    # Calculate overall metrics
    overall_metrics = {
        "validation_loss": np.mean(all_losses),
        "validation_iou": np.mean(all_ious),
        "num_samples": len(all_predictions),
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "config": config
    }
    
    print(f"Validation completed:")
    print(f"  - Average Loss: {overall_metrics['validation_loss']:.4f}")
    print(f"  - Average IoU: {overall_metrics['validation_iou']:.4f}")
    print(f"  - Number of samples: {overall_metrics['num_samples']}")
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("logs") / "validation_results" / timestamp
    
    # Save results
    save_validation_results(
        predictions=all_predictions,
        targets=all_targets,
        metrics=overall_metrics,
        save_dir=output_dir,
        filenames=[Path(f) for f in filenames]
    )
    
    return overall_metrics

if __name__ == "__main__":
    try:
        metrics = run_validation()
        print("Validation completed successfully!")
    except Exception as e:
        print(f"Error during validation: {e}")
        raise
