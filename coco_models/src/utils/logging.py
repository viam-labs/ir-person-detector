import logging
import sys
from pathlib import Path
from omegaconf import DictConfig

def setup_logging(cfg: DictConfig):
    """
    Setup logging configuration
    """
    log_dir = Path(cfg.save_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__) 