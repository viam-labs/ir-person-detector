import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import hydra
from omegaconf import DictConfig

class FasterRCNNDetector(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(FasterRCNNDetector, self).__init__()
        
        # Load pretrained model with default weights
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        
        #single class detection, must define number of classes from cfg 
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, cfg.model.num_classes + 1  )
        
        self.transforms = weights.transforms()
    
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        images = [self.transforms(img) for img in images]
        
        # Forward pass
        if self.training:
            return self.model(images, targets)
        else:
            return self.model(images)

@hydra.main(config_path="../../configs", config_name="model/faster_rcnn", version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.model.device if torch.cuda.is_available() else 'cpu')
    
    model = FasterRCNNDetector(cfg).to(device)
    print("fasterrcnn model created and moved to device")
    
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
    print("\nLoss function and optimizer initialized for fasterrcnn detector")

if __name__ == "__main__":
    main() 