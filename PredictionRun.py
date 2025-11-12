import torch
import numpy as np
import os
import glob
from PIL import Image
import toml
from NetModule import NetModule
import torch.nn.functional as F
from pathlib import Path


def load_config(config_path="config.toml"):
    """Load configuration from TOML file"""
    with open(config_path, 'r') as f:
        config = toml.load(f)
    return config


def load_images_from_folder(folder_path):
    """Load all PNG images from the specified folder"""
    image_files = []
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    image_files.sort()  # Sort to ensure consistent ordering
    print(f"Found {len(image_files)} images in {folder_path}")
    return image_files


def preprocess_image(image_path, target_size=(480, 288)):
    """Load and preprocess a single image for model inference"""
    # Load image using PIL and convert to grayscale
    img = Image.open(image_path).convert('L')
    
    # Resize to target size
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor


def load_model_from_checkpoint(checkpoint_path, config):
    """Load the trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load model from checkpoint
    model = NetModule.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()  # Set to evaluation mode
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model loaded on device: {device}")
    
    return model, device


def run_prediction(model, image_tensor, device):
    """Run model prediction on a single image"""
    with torch.no_grad():
        # Move image to device
        image_tensor = image_tensor.to(device)
        
        # Forward pass
        logits = model(image_tensor)
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=1)
        
        # Get predicted class (argmax)
        predicted_class = torch.argmax(probabilities, dim=1)
        
        return probabilities.cpu(), predicted_class.cpu()


def save_predictions(predictions, predicted_classes, output_folder, filename):
    """Save prediction results to files"""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    base_name = Path(filename).stem
    
    class_mask = predicted_classes.squeeze().numpy().astype(np.uint8)
    # Save colored prediction mask for visualization
    colored_mask = create_colored_mask(class_mask)
    colored_path = os.path.join(output_folder, f"{base_name}_colored.png")
    Image.fromarray(colored_mask).save(colored_path)
    
    print(f"Saved predictions for {base_name}")
    

def create_colored_mask(mask, num_classes=12):
    """Create a colored visualization of the prediction mask"""
    # Create a color map for different classes
    colors = np.array([
        [0, 0, 0],       # Class 0: Black
        [255, 0, 0],     # Class 1: Red
        [0, 255, 0],     # Class 2: Green
        [0, 0, 255],     # Class 3: Blue
        [255, 255, 0],   # Class 4: Yellow
        [255, 0, 255],   # Class 5: Magenta
        [0, 255, 255],   # Class 6: Cyan
        [128, 0, 0],     # Class 7: Dark Red
        [0, 128, 0],     # Class 8: Dark Green
        [0, 0, 128],     # Class 9: Dark Blue
        [128, 128, 0],   # Class 10: Olive
        [128, 0, 128],   # Class 11: Purple
    ], dtype=np.uint8)
    
    # Ensure we don't exceed color array bounds
    mask = np.clip(mask, 0, num_classes - 1)
    
    # Map mask values to colors
    colored_mask = colors[mask]
    
    return colored_mask


def main():
    """Main function to run the prediction pipeline"""
    # Configuration
    config = load_config()
    images_folder = "./data/images"
    # get the latest checkpoint in the logs folder
    logs_folder = "./logs/layer_segmentation"
    checkpoint_files = glob.glob(os.path.join(logs_folder, "*.ckpt"))
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)  # Sort by modification time
    checkpoint_path = checkpoint_files[0] if checkpoint_files else None
    output_folder = "./logs/predicted_out"
    
    # Step 1: Load images from folder
    print("Step 1: Loading images from folder...")
    image_files = load_images_from_folder(images_folder)
    
    if not image_files:
        print("No images found in the specified folder!")
        return
    
    # Step 2: Load model from checkpoint
    print("Step 2: Loading model from checkpoint...")
    model, device = load_model_from_checkpoint(checkpoint_path, config)
    
    # Step 3: Process each image
    print("Step 3: Running predictions on images...")
    for i, image_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Preprocess image
        image_tensor = preprocess_image(image_path, target_size=(480, 288))
        
        # Run prediction
        probabilities, predicted_classes = run_prediction(model, image_tensor, device)
        
        # Step 4: Save results
        save_predictions(probabilities, predicted_classes, output_folder, image_path)
    
    print(f"\nPrediction completed! Results saved to: {output_folder}")


if __name__ == "__main__":
    main()
