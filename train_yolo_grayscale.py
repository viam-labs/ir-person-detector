#use this to train the model on grayscale images if it is performing poorly on rgb images

from ultralytics import YOLO
import torch
import torch.nn as nn

def modify_yolo_for_grayscale(model):
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

if __name__ == "__main__":
    # Check for MPS (Apple Silicon) or CUDA
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("Using CUDA")
    else:
        device = 'cpu'
        print("Using CPU")

    # Load the base model
    model = YOLO('yolov8n.pt')
    
    # Modify for grayscale input
    model = modify_yolo_for_grayscale(model)
    
    # Train the model
    model.train(
        data='data/thermal/flir.yaml',
        epochs=50,
        imgsz=640,
        batch=64,
        device=device,
        workers=0, 
        name='yolov8n_flir_grayscale'
    ) 