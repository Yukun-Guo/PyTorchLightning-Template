"""
Example script to load and analyze saved validation results.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_validation_results(results_dir):
    """Load validation results from a timestamped directory."""
    results_path = Path(results_dir)
    
    # Load metrics
    metrics_file = results_path / "validation_metrics.json"
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Load predictions and ground truth
    predictions_dir = results_path / "predictions"
    prediction_files = sorted(predictions_dir.glob("*_prediction.npy"))
    ground_truth_files = sorted(predictions_dir.glob("*_ground_truth.npy"))
    
    predictions = []
    ground_truths = []
    
    for pred_file, gt_file in zip(prediction_files, ground_truth_files):
        predictions.append(np.load(pred_file))
        ground_truths.append(np.load(gt_file))
    
    return metrics, predictions, ground_truths

def analyze_sample_prediction(predictions, ground_truths, sample_idx=0):
    """Analyze a specific sample prediction."""
    pred = predictions[sample_idx]
    gt = ground_truths[sample_idx]
    
    print(f"Sample {sample_idx}:")
    print(f"  Prediction shape: {pred.shape}")
    print(f"  Ground truth shape: {gt.shape}")
    print(f"  Prediction classes range: {pred.argmax(axis=0).min()} - {pred.argmax(axis=0).max()}")
    print(f"  Ground truth classes range: {gt.min()} - {gt.max()}")
    
    # Convert prediction probabilities to class predictions
    pred_classes = pred.argmax(axis=0)
    
    # Calculate per-sample metrics
    correct_pixels = (pred_classes == gt).sum()
    total_pixels = gt.size
    accuracy = correct_pixels / total_pixels
    
    print(f"  Pixel accuracy: {accuracy:.4f}")
    
    return pred_classes, gt

def main():
    """Main function to demonstrate loading and analyzing results."""
    # Find the most recent validation results
    results_base_dir = Path("logs/validation_results")
    if not results_base_dir.exists():
        print("No validation results found. Please run PredictionVal.py first.")
        return
    
    # Get the most recent results directory
    results_dirs = [d for d in results_base_dir.iterdir() if d.is_dir()]
    if not results_dirs:
        print("No validation results found. Please run PredictionVal.py first.")
        return
    
    latest_results_dir = max(results_dirs, key=lambda x: x.name)
    print(f"Loading results from: {latest_results_dir}")
    
    # Load results
    metrics, predictions, ground_truths = load_validation_results(latest_results_dir)
    
    # Print overall metrics
    print("\nOverall Validation Metrics:")
    print(f"  Average Loss: {metrics['validation_loss']:.4f}")
    print(f"  Average IoU: {metrics['validation_iou']:.4f}")
    print(f"  Number of samples: {metrics['num_samples']}")
    print(f"  Timestamp: {metrics['timestamp']}")
    print(f"  Model used: {metrics['model_path']}")
    
    # Analyze first few samples
    print(f"\nAnalyzing first 3 samples:")
    for i in range(min(3, len(predictions))):
        pred_classes, gt = analyze_sample_prediction(predictions, ground_truths, i)
    
    print(f"\nValidation results successfully loaded and analyzed!")
    print(f"You can use the loaded data for further analysis:")
    print(f"  - metrics: dictionary with overall validation metrics")
    print(f"  - predictions: list of {len(predictions)} prediction arrays")
    print(f"  - ground_truths: list of {len(ground_truths)} ground truth arrays")

if __name__ == "__main__":
    main()