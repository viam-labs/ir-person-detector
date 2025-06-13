import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_Weights
import math
import hydra
from omegaconf import DictConfig

class EfficientNetDetector(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(EfficientNetDetector, self).__init__()
        
        # default weights loaded in 
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Get the number of features from the backbone
        self.num_features = 1280  # EfficientNet-B0 features
        
        # Store expected input size
        self.expected_size = cfg.dataset.image.size
        
        # Detection heads
        self.bbox_head = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)  # bounding box (x, y, w, h)
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, cfg.model.num_classes)  # Number of classes
        )
        
        # Loss function
        self.criterion = DetectionLoss(cfg)
    
    def forward(self, x):
        # Verify input size
        _, _, h, w = x.shape
        exp_h, exp_w = self.expected_size
        if h != exp_h or w != exp_w:
            raise ValueError(f"Expected input size {self.expected_size}, got {(h, w)}")
            
        # Extract features from backbone
        features = self.backbone(x)  # Shape: [batch_size, 1280]
        
        # Get bounding box predictions
        bbox_pred = self.bbox_head(features)
        
        # Get class predictions
        cls_pred = self.cls_head(features)
        
        return bbox_pred, cls_pred
    
    def compute_loss(self, output, targets):
        bbox_pred, cls_pred = output
        batch_size = bbox_pred.shape[0]
        device = bbox_pred.device
        
        # Initialize tensors to store all boxes and classes
        all_boxes = []
        all_classes = []
        
        # Process each image's targets
        for i in range(batch_size):
            boxes = targets['boxes'][i]  # This is already a tensor
            # Create one-hot encoded class labels (assuming binary classification)
            classes = torch.ones(boxes.shape[0], 1, device=device)
            
            all_boxes.append(boxes)
            all_classes.append(classes)
        
        # Concatenate all boxes and classes
        bbox_target = torch.cat(all_boxes, dim=0)
        cls_target = torch.cat(all_classes, dim=0)
        
        # Repeat predictions for each target box
        bbox_pred_repeated = []
        cls_pred_repeated = []
        for i in range(batch_size):
            num_boxes = targets['boxes'][i].shape[0]
            bbox_pred_repeated.append(bbox_pred[i:i+1].repeat(num_boxes, 1))
            cls_pred_repeated.append(cls_pred[i:i+1].repeat(num_boxes, 1))
        
        bbox_pred = torch.cat(bbox_pred_repeated, dim=0)
        cls_pred = torch.cat(cls_pred_repeated, dim=0)
        
        # Compute loss
        return self.criterion(bbox_pred, cls_pred, bbox_target, cls_target)

class DetectionLoss(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(DetectionLoss, self).__init__()
        self.lambda_box = cfg.loss.box_loss_weight
        self.lambda_cls = cfg.loss.cls_loss_weight
        self.bbox_loss = nn.SmoothL1Loss()
        self.cls_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, bbox_pred, cls_pred, bbox_target, cls_target):
        # Calculate bounding box loss
        box_loss = self.bbox_loss(bbox_pred, bbox_target)
        
        # Calculate classification loss
        cls_loss = self.cls_loss(cls_pred, cls_target)
        
        # Combine losses
        total_loss = self.lambda_box * box_loss + self.lambda_cls * cls_loss
        
        return total_loss, box_loss, cls_loss

@hydra.main(config_path="../../configs", config_name="model/effnet", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.model.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = EfficientNetDetector(cfg).to(device)
    print("effnet model created and moved to device")
    
    # Create loss function
    criterion = DetectionLoss(cfg)
    
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
    print("\nLoss function and optimizer initialized for effnet detector")
    
    # Test with dummy input
    dummy_input = torch.randn(1, cfg.model.input_channels, 640, 640).to(device)
    bbox_pred, cls_pred = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Bounding box predictions shape: {bbox_pred.shape}")
    print(f"Classification predictions shape: {cls_pred.shape}")
    
    # Print model summary using a dummy input
    print("\nModel Summary:")
    print(model)

if __name__ == "__main__":
    main() 