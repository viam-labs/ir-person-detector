from ultralytics import YOLO
import torch

if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = 'cpu'
    print("using CPU")

# Load the model
model = YOLO('yolov8n.pt') 

print("\nTraining Configuration:")
print(f"Device: {device}")
print(f"Model: yolov8n.pt")
print(f"Dataset: data/thermal/flir.yaml")
print(f"Image size: 640x640")
print(f"Batch size: 64")
print(f"Epochs: 50")

# Train the model
model.train(
    data='data/thermal/flir.yaml',
    epochs=50,
    imgsz=640,
    batch=32,
    device=device,
    workers=4,
    name='yolov8n_flir_gpu',

    # the following are all set to their default for now, can change if needed
    hsv_h = 0.015, #hue
    hsv_s = 0.7, #increasing saturation variation to distinguish between objects in IR images
    hsv_v = 0.4,#value/brightness increase
    mosaic = 1.0, #cuts 4 images into 4 and combines them into single images, helps with varying thermal patterns 

    #save best model
    save = True,
    save_period = 10
)
