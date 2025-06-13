#set up to work with the original flir dataset in COCO format
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import hydra
from omegaconf import DictConfig

class ThermalDetector(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ThermalDetector, self).__init__()
        
        # Get configuration values
        backbone_channels = cfg.model.backbone.channels
        kernel_size = cfg.model.backbone.kernel_size
        padding = cfg.model.backbone.padding
        hidden_size = cfg.model.detector.hidden_size
        dropout = cfg.model.detector.dropout
        input_channels = cfg.model.input_channels
        output_size = cfg.model.output_size
        
        # CNN backbone
        layers = []
        for out_channels in backbone_channels:
            layers.extend([
                nn.Conv2d(input_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            input_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Calculate the size of flattened features
        # Input size of 640x640, after 3 maxpool layers (2x2) it becomes 80x80
        flattened_size = backbone_channels[-1] * 80 * 80
        
        # Detection head
        self.detector = nn.Sequential(
            nn.Linear(flattened_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)  # 4 for bbox + 1 for class
        )
        
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Flatten
        features = features.view(features.size(0), -1)
        
        # Get predictions
        predictions = self.detector(features)
        
        # Split into bbox and class predictions
        bbox_pred = predictions[:, :4]
        cls_pred = predictions[:, 4:]
        
        return bbox_pred, cls_pred

@hydra.main(config_path="../../configs", config_name="model/custom_detector", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.model.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = ThermalDetector(cfg).to(device)
    print("custom detector model created and moved to device")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    print("\nLoss function and optimizer initialized for custom detector")

if __name__ == "__main__":
    main() 