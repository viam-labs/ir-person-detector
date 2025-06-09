import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import efficientnet_b0
import math

class EfficientNetDetector(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(EfficientNetDetector, self).__init__()
        
        # load pretrained EfficientNet-B0
        self.backbone = efficientnet_b0(pretrained=pretrained)
        
        self.backbone.classifier = nn.Identity()
        
        # Get the number of features from the backbone
        self.num_features = self.backbone.classifier[1].in_features
        
        # Detection heads
        self.bbox_head = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)  # 4 values for bounding box (x, y, w, h)
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)  # 1 class for person
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Get bounding box predictions
        bbox_pred = self.bbox_head(features)
        
        # Get class predictions
        cls_pred = self.cls_head(features)
        
        return bbox_pred, cls_pred

class DetectionLoss(nn.Module):
    def __init__(self, lambda_box=1.0, lambda_cls=1.0):
        super(DetectionLoss, self).__init__()
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
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

def create_model(device='cuda'):
    
    model = EfficientNetDetector(num_classes=1, pretrained=True)
    
    model = model.to(device)
    
    criterion = DetectionLoss(lambda_box=1.0, lambda_cls=1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    return model, criterion, optimizer, scheduler

if __name__ == "__main__":
    # running model on dummy input to check if it is working
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, criterion, optimizer, scheduler = create_model(device)
  
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    
    # Forward pass
    bbox_pred, cls_pred = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Bounding box predictions shape: {bbox_pred.shape}")
    print(f"Classification predictions shape: {cls_pred.shape}")
    
    # Print model summary
    print("\nModel Summary:")
    print(model) 