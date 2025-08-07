from torch.utils.data import DataLoader
from datasets.ir_dataset import IRDataset
from datasets.flir_dataset import FLIRDataset
from utils.transforms import GPUCollate
from utils.transforms import build_transforms
from omegaconf import DictConfig
import hydra
import torch
from pathlib import Path

def compute_mean_std_pytorch(data_loader, num_batches=100):
    n = 0
    mean = 0.0
    M2 = 0.0

    for i, (images, _) in enumerate(data_loader):
        # images: (batch, channels, H, W)
        images = images.float()
        if images.max() > 1.0:
            images = images / 255.0  

        batch_pixels = images.numel()
        batch_mean = images.mean().item()
        batch_var = images.var(unbiased=False).item()

        # running mean and var (Welford's algorithm)
        delta = batch_mean - mean
        total_n = n + batch_pixels
        mean += delta * batch_pixels / total_n
        M2 += batch_var * batch_pixels + delta**2 * n * batch_pixels / total_n
        n = total_n

        if i + 1 >= num_batches:
            break

    std = (M2 / n) ** 0.5
    return mean, std

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    #creating dataset and dataloader 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    train_dataset = FLIRDataset(
        json_file=Path(cfg.dataset.data.train_annotations),
        thermal_dir=Path(cfg.dataset.data.train_images),
    )
    train_transform = build_transforms(cfg, is_train=True, test=False) #commented out transforms to avoid changing the dataset before calculating mean and std
    loader = DataLoader(
    train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers= 0,
        pin_memory= cfg.training.pin_memory,
        collate_fn=GPUCollate(device, train_transform) 
        )

    mean, std = compute_mean_std_pytorch(loader)
    print(f"Mean: {mean:.4f}, Std: {std:.4f}")

if __name__ == "__main__":
    main() 