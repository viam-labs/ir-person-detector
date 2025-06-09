import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

def get_ssdlite_model(cfg): #using config from hydra
    """
    ssd lite model from torchvision
    """
    # Load pretrained model
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights)
    
    # Modify for single class (person)
    num_classes = 1 # Background + Person
    in_channels = 256  # Number of input channels for the classifier
    
    model.head.classifier = torchvision.models.detection.ssd.SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=model.anchor_generator.num_anchors_per_location(),
        num_classes=num_classes
    )
    
    # Move model to device
    device = torch.device(cfg.model.device)
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=cfg.training.early_stopping_patience,
        verbose=True
    )
    
    return model, optimizer, scheduler

if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy config
    class DummyConfig:
        def __init__(self):
            self.model = type('obj', (object,), {
                'device': device
            })
            self.training = type('obj', (object,), {
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'early_stopping_patience': 5
            })
    
    cfg = DummyConfig()
    
    # Create model
    model, optimizer, scheduler = get_ssdlite_model(cfg)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 320, 320).to(device)
    
    # Forward pass
    model.train()
    predictions = model([dummy_input])
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Predictions: {predictions}")
    
    # Print model summary
    print("\nModel Summary:")
    print(model) 