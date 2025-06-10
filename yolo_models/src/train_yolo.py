import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
from ultralytics import YOLO
import torch
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = 'cpu'
    print("No GPU available, using CPU")

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Setup logging
    log.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seed
    torch.manual_seed(cfg.experiment.seed)
    
    # Initialize YOLO model
    model = YOLO(f"yolov8{cfg.model.version}.pt")
    
    # Train the model
    results = model.train(
        data=cfg.dataset.data.train,  # Path to training data is from the dataset config 
        epochs=cfg.model.epochs,
        imgsz=cfg.model.img_size,
        batch=cfg.model.batch_size,
        device=cfg.model.device,
        workers=cfg.training.num_workers,
        project=cfg.logging.save_dir,
        name=f"yolo_{cfg.model.version}",
        exist_ok=True,
        pretrained=cfg.model.pretrained,
        optimizer=cfg.model.optimizer,
        lr0=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        momentum=cfg.model.momentum,
        warmup_epochs=cfg.model.warmup_epochs,
        warmup_momentum=cfg.model.warmup_momentum,
        warmup_bias_lr=cfg.model.warmup_bias_lr,
        box=cfg.model.box,
        cls=cfg.model.cls,
        dfl=cfg.model.dfl,
        pose=cfg.model.pose,
        kobj=cfg.model.kobj,
        label_smoothing=cfg.model.label_smoothing,
        nbs=cfg.model.nbs,
        overlap_mask=cfg.model.overlap_mask,
        mask_ratio=cfg.model.mask_ratio,
        dropout=cfg.model.dropout,
        val=cfg.model.val,
        plots=cfg.model.plots
    )
    
    # Save the final model
    model.save(f"{cfg.logging.save_dir}/yolo_{cfg.model.version}/weights/best.pt")

if __name__ == "__main__":
    main()
