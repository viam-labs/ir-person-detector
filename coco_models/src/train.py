#trainign script for all models except yolo
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from tqdm import tqdm
import wandb

# Import your model implementations
from models.custom_detector import ThermalDetector
from models.faster_rcnn_detector import FasterRCNNDetector
from models.effnet_detector import EfficientNetDetector
from models.ssdlite_detector import SSDLiteDetector

# Import dataset
from datasets.flir_dataset import FLIRDataset

# Import logging utilities
from utils.logging import setup_logging

log = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, optimizer, device, cfg: DictConfig):
    """
    Train the model
    """
    # Initialize wandb if enabled
    if cfg.logging.wandb.enabled:
        wandb.init(
            project=cfg.logging.wandb.project,
            config=cfg,
            tags=[cfg.model.name]
        )
        wandb.watch(model)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(cfg.training.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.training.num_epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = model.compute_loss(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item()})
            
            # Log to wandb
            if cfg.logging.wandb.enabled and batch_idx % cfg.logging.log_interval == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': optimizer.param_groups[0]['lr']
                })
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{cfg.training.num_epochs} [Val]')
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = model.compute_loss(output, target)
                val_loss += loss.item()
                
                # Update progress bar
                val_pbar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Log epoch metrics
        log.info(f'Epoch {epoch+1}/{cfg.training.num_epochs}: '
                f'Train Loss: {avg_train_loss:.4f}, '
                f'Val Loss: {avg_val_loss:.4f}')
        
        if cfg.logging.wandb.enabled:
            wandb.log({
                'epoch': epoch + 1,
                'train/epoch_loss': avg_train_loss,
                'val/epoch_loss': avg_val_loss
            })
        
        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint_path = Path(cfg.logging.save_dir) / 'best_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            
            log.info(f'Saved best model checkpoint to {checkpoint_path}')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= cfg.training.early_stopping_patience:
            log.info(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    if cfg.logging.wandb.enabled:
        wandb.finish()

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Setup logging
    setup_logging(cfg.logging)
    log.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")
    
    torch.manual_seed(cfg.experiment.seed)
    
    # Create model based on configuration
    if cfg.model.name == "custom_detector":
        model = ThermalDetector(
            input_channels=cfg.model.input_channels,
            output_size=cfg.model.output_size,
            backbone_channels=cfg.model.backbone.channels,
            kernel_size=cfg.model.backbone.kernel_size,
            padding=cfg.model.backbone.padding,
            hidden_size=cfg.model.detector.hidden_size,
            dropout=cfg.model.detector.dropout
        )
    elif cfg.model.name == "faster_rcnn":
        model = FasterRCNNDetector(cfg)
    elif cfg.model.name == "effnet":
        model = EfficientNetDetector(cfg)
    elif cfg.model.name == "ssdlite":
        model = SSDLiteDetector(cfg)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.name}")
    
    # Get device from model 
    device = next(model.parameters()).device
    
    # Create datasets
    train_dataset = FLIRDataset(
        json_file=Path(cfg.data.train).parent / "coco.json",
        thermal_dir=Path(cfg.data.train),
        transform=cfg.dataset.transform
    )
    
    val_dataset = FLIRDataset(
        json_file=Path(cfg.data.val).parent / "coco.json",
        thermal_dir=Path(cfg.data.val),
        transform=cfg.dataset.transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory
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