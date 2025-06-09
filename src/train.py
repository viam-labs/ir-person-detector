#trainign script for all models except yolo
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import logging
from pathlib import Path

# Import your model implementations
from models.custom_detector import SimpleThermalDetector
from models.faster_rcnn import get_faster_rcnn_model
from models.efficientnet import get_efficientnet_model
from models.ssdlite import get_ssdlite_model

# Import dataset
from datasets.flir_dataset import FLIRDataset

# Import training utilities
from utils.training import train_model
from utils.logging import setup_logging

log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Setup logging
    setup_logging(cfg.logging)
    log.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seed
    torch.manual_seed(cfg.experiment.seed)
    
    # Create model based on configuration
    if cfg.model.type == "custom":
        model = SimpleThermalDetector(
            backbone_channels=cfg.model.backbone.channels,
            kernel_size=cfg.model.backbone.kernel_size,
            padding=cfg.model.backbone.padding,
            hidden_size=cfg.model.detector.hidden_size,
            dropout=cfg.model.detector.dropout
        )
    elif cfg.model.type == "faster_rcnn":
        model = get_faster_rcnn_model(cfg.model)
    elif cfg.model.type == "efficientnet":
        model = get_efficientnet_model(cfg.model)
    elif cfg.model.type == "ssdlite":
        model = get_ssdlite_model(cfg.model)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    
    # Move model to device
    device = torch.device(cfg.experiment.device)
    model = model.to(device)
    
    # Create datasets
    train_dataset = FLIRDataset(
        json_file=Path(cfg.dataset.train_path) / "coco.json",
        thermal_dir=Path(cfg.dataset.train_path) / "data",
        transform=cfg.dataset.transform
    )
    
    val_dataset = FLIRDataset(
        json_file=Path(cfg.dataset.val_path) / "coco.json",
        thermal_dir=Path(cfg.dataset.val_path) / "data",
        transform=cfg.dataset.transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        cfg=cfg
    )

if __name__ == "__main__":
    main() 