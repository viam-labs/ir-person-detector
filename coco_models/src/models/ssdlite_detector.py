import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
import hydra
from omegaconf import DictConfig

class SSDLiteDetector(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(SSDLiteDetector, self).__init__()
        
        # Load pretrained model with default weights
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=weights)
        
        # Replace the classifier for single-class detection
        in_channels = self.model.head.classifier.module_list[0].in_channels
        self.model.head.classifier = torchvision.models.detection.ssd.SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=self.model.anchor_generator.num_anchors_per_location(),
            num_classes=cfg.model.num_classes + 1  # +1 for background
        )
        
        # Store the transforms for preprocessing
        self.transforms = weights.transforms()
    
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        # Apply transforms
        images = [self.transforms(img) for img in images]
        
        # Forward pass
        if self.training:
            return self.model(images, targets)
        else:
            return self.model(images)

@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg.model.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SSDLiteDetector(cfg).to(device)
    print("ssdlite model created and moved to device")

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
    print("\nLoss function and optimizer initialized for ssdlite detector")

if __name__ == "__main__":
    main() 