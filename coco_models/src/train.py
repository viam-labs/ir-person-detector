#trainign script for all models except yolo
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
from models.custom_detector import ThermalDetector
from models.faster_rcnn_detector import FasterRCNNDetector
from models.effnet_detector import EfficientNetDetector
from models.ssdlite_detector import SSDLiteDetector
from datasets.ir_dataset import IRDataset
from utils.transforms import build_transforms, GPUCollate
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import gc
from omegaconf import OmegaConf
log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, cfg: DictConfig):
    # Create tensorboard writer using Hydra's output directory
    writer = SummaryWriter(Path(cfg.logging.save_dir) / 'tensorboard')  # Use Hydra's output directory
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(cfg.training.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_losses = {'loss_classifier': 0.0,
                        'loss_box_reg': 0.0,
                        'loss_objectness': 0.0,
                        'loss_rpn_box_reg': 0.0}
                        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.training.num_epochs} [Train]')
        for batch_idx, (data, targets) in enumerate(train_pbar):
            optimizer.zero_grad()

            if cfg.model.name in ["faster_rcnn", "ssdlite"]: #for torchvision models: they return a Dict[Tensor] which contains classification and regression losses
                loss_dict = model(data, targets)
                loss = (
                loss_dict['loss_classifier'] * cfg.training.loss.cls_loss_weight +
                loss_dict['loss_box_reg'] * cfg.training.loss.box_loss_weight +
                loss_dict['loss_objectness'] * cfg.training.loss.rpn_loss_weight +
                loss_dict['loss_rpn_box_reg'] * cfg.training.loss.rpn_box_reg_loss_weight
            ) #sum of classification and regression losses according to weights 

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping to avoid exploding gradients
            optimizer.step()
            train_loss += loss.item() #maintain running loss total 

            # Log individual losses per batch to tensorboard
            for loss_name, loss_value in loss_dict.items():
                train_losses[loss_name] += loss_value.item()
                writer.add_scalar(f'BatchLoss/{loss_name}', loss_value.item(), 
                                epoch * len(train_loader) + batch_idx)

                # Update progress bar
            train_pbar.set_postfix({'train loss this batch': loss.item()})

        avg_losses = {k: v/len(train_loader) for k, v in train_losses.items()}
        avg_train_loss = sum(avg_losses.values())
        avg_train_loss = train_loss / len(train_loader) 
        # Choose validation function based on model type
        if cfg.model.name in ["faster_rcnn", "ssdlite"]:
            avg_val_loss = evaluate_validation(model, val_loader, device, epoch, cfg)

        scheduler.step(avg_val_loss)

        # Log epoch metrics to tensorboard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        # Log learning rate
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log epoch metrics
        log.info(f'Epoch {epoch+1}/{cfg.training.num_epochs}: '
                f'Avg Train Loss: {avg_train_loss:.4f}, '
                f'Avg Val Loss: {avg_val_loss:.4f}')

        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = Path(cfg.logging.save_dir) / 'best_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            patience_counter = 0
        else:
             patience_counter += 1
        
        #early stopping
        if patience_counter >= cfg.training.early_stopping_patience: #10 epochs without improvement 
            log.info(f'Early stopping triggered after {epoch + 1} epochs')
            break

    writer.close()
    return best_val_loss  # for optuna to minimize the validation loss 

def evaluate_validation(model, val_loader, device, epoch, cfg: DictConfig):
    
    was_training = model.training #boolean flag to return model to mode it was in before evaluation

    # force loss-returning behavior
    model.train()

    # workaround to freeze batchnorm and dropout layers to aviod training them
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.Dropout)):
            m.eval()

    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            loss_dict = model(images, targets)
            
            batch_loss = sum(loss_dict.values()).item()
            val_loss += batch_loss
    # back to train mode 
    model.train(was_training)
    
    return val_loss / len(val_loader)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    # Log config
    log.info(f"check device: {torch.cuda.is_available()}")
    log.info(f"config: \n{OmegaConf.to_yaml(cfg)}")
    
    torch.manual_seed(cfg.experiment.seed)
    device = cfg.model.device
    log.info(f"device is: {device}")
    # model created based on config
    if cfg.model.name == "custom_detector":
        model = ThermalDetector(cfg).to(device)
        log.info("custom detector model created and moved to device")
    elif cfg.model.name == "faster_rcnn":
        model = FasterRCNNDetector(cfg).to(device)
        log.info("faster rcnn model created and moved to device")
    elif cfg.model.name == "effnet":
        model = EfficientNetDetector(cfg).to(device)
        log.info("efficientnet model created and moved to device")
    elif cfg.model.name == "ssdlite":
        model = SSDLiteDetector(cfg).to(device)
        log.info("ssdlite model created and moved to device")
        summary(model, (32, 1, 320, 320))
    else:
        raise ValueError(f"Unknown model type: {cfg.model.name}")
   

    train_transform = build_transforms(cfg, is_train=True, test=False) #investigate whether each image can be transformed differently
    val_transform = build_transforms(cfg, is_train=False, test=False)    

    train_dataset = IRDataset(
        json_file=Path(cfg.dataset.data.train_annotations),
        thermal_dir=Path(cfg.dataset.data.train_images),
    )

    val_dataset = IRDataset(
        json_file=Path(cfg.dataset.data.val_annotations),
        thermal_dir=Path(cfg.dataset.data.val_images),
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers= cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        collate_fn=GPUCollate(device, train_transform) 
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers= cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        collate_fn=GPUCollate(device, val_transform)
    )
    
    # Create optimizer
  
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay
        )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3, 
        min_lr=1e-6,
        verbose=True 
    )
    
    # train model and get best validation loss
    best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        cfg=cfg
    )
    gc.collect() #to avoid CUDA out of memory error on optuna
    torch.cuda.empty_cache()
    # best validation loss for Optuna 
    return best_val_loss

if __name__ == "__main__":
    main() 