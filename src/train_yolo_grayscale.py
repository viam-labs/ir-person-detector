#use this to train the model on grayscale images if it is performing poorly on rgb images

from ultralytics import YOLO
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig

def convert_to_grayscale(model):
    """
    Modify YOLO's first layer to accept grayscale input (1 channel instead of 3)
    """
    # Get the first convolutional layer
    first_conv = model.model.model[0]
    
    # Create new conv layer with 1 input channel
    new_conv = nn.Conv2d(
        in_channels=1,  # Changed from 3 to 1
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None
    )
    
    # initialize the new layer's weights
    # Average the weights across the RGB channels
    with torch.no_grad():
        new_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
        if first_conv.bias is not None:
            new_conv.bias.data = first_conv.bias.data
    
    # Replace the first layer
    model.model.model[0] = new_conv
    
    return model

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Check for MPS (Apple Silicon) or CUDA
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = 'cpu'
        print("Using CPU")

    # Load the base model
    model = YOLO(f"yolov8{cfg.model.version}.pt")
    
    # Modify for grayscale input
    model = convert_to_grayscale(model)
    
    print("\nTraining Configuration:")
    print(f"Device: {device}")
    print(f"Model: yolov8{cfg.model.version}.pt")
    print(f"Dataset: {cfg.data.train}")
    print(f"Image size: {cfg.model.img_size}")
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Epochs: {cfg.training.num_epochs}")
    
    # Train the model
    model.train(
        data=cfg.data.train,  # Use path from dataset config
        epochs=cfg.training.num_epochs,
        imgsz=cfg.model.img_size,
        batch=cfg.training.batch_size,
        device=device,
        workers=cfg.training.num_workers,
        name=f"yolov8{cfg.model.version}_flir_grayscale",
        
        # Augmentation settings from dataset config
        hsv_h=cfg.dataset.augmentation.hsv_h,
        hsv_s=cfg.dataset.augmentation.hsv_s,
        hsv_v=cfg.dataset.augmentation.hsv_v,
        degrees=cfg.dataset.augmentation.degrees,
        translate=cfg.dataset.augmentation.translate,
        scale=cfg.dataset.augmentation.scale,
        shear=cfg.dataset.augmentation.shear,
        perspective=cfg.dataset.augmentation.perspective,
        flipud=cfg.dataset.augmentation.flipud,
        fliplr=cfg.dataset.augmentation.fliplr,
        mosaic=cfg.dataset.augmentation.mosaic,
        mixup=cfg.dataset.augmentation.mixup,
        
        # Training settings from model config
        save=True,
        save_period=cfg.model.save_period
    )

if __name__ == "__main__":
    main() 