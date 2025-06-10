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

@hydra.main(config_path="../configs", config_name="config", version_base=None) #setup from hydra to load config file 
def main(cfg: DictConfig):
    # Get the script's directory
    script_dir = Path(__file__).parent.parent
    # Override config with YOLO-specific settings using absolute path
    yolo_config_path = script_dir / "configs" / "model" / "yolo.yaml"
    cfg = OmegaConf.merge(cfg, OmegaConf.load(str(yolo_config_path)))
    
    # Setup logging
    log.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seed
    torch.manual_seed(cfg.experiment.seed)
    
    # Initialize YOLO model
    model = YOLO(f"yolov8{cfg.model.version}.pt")
    
    # Train the model
    results = model.train( 
        data=cfg.data,
        epochs=cfg.model.epochs,
        imgsz=cfg.model.img_size,
        batch=cfg.model.batch_size,
        device=cfg.model.device,
        workers=cfg.dataset.num_workers,
        project=cfg.logging.save_dir,
        name=f"yolo_{cfg.model.version}",
        exist_ok=True,
        pretrained=cfg.model.pretrained,
        optimizer="Adam",
        lr0=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        momentum=cfg.training.momentum,
        warmup_epochs=cfg.training.warmup_epochs,
        warmup_momentum=cfg.training.warmup_momentum,
        warmup_bias_lr=cfg.training.warmup_bias_lr,
        box=cfg.training.box,
        cls=cfg.training.cls,
        dfl=cfg.training.dfl,
        pose=cfg.training.pose,
        kobj=cfg.training.kobj,
        label_smoothing=cfg.training.label_smoothing,
        nbs=cfg.training.nbs,
        overlap_mask=cfg.training.overlap_mask,
        mask_ratio=cfg.training.mask_ratio,
        dropout=cfg.training.dropout,
        val=cfg.training.val,
        plots=cfg.training.plots
    )
    
    # Save the final model
    model.save(f"{cfg.logging.save_dir}/yolo_{cfg.model.version}/weights/best.pt")

if __name__ == "__main__":
    main()
