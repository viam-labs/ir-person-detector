# used to format dataset from yolo format to pytorch format for training 

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from torchvision import transforms

class FLIRDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((640, 640)), #resizing from 640x512 to 640x640
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]) #normalizing image 
        ])
        
        # getting all image files
        self.image_files = [f for f in os.listdir(image_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # loading image 
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path) 
        
        # Load label
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        # targets for bounding box and class
        bbox_target = torch.zeros(4)  # [x, y, w, h]
        cls_target = torch.zeros(1)   # 1 for person
        
        # reading label file if it exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    # YOLO format: class x_center y_center width height
                    cls, x, y, w, h = map(float, line.strip().split())
                    
                    # Convert to absolute coordinates
                    img_w, img_h = image.size
                    bbox_target = torch.tensor([
                        x * img_w,  # x_center
                        y * img_h,  # y_center
                        w * img_w,  # width
                        h * img_h   # height
                    ])
                    
                    # Set class (1 for person)
                    cls_target[0] = 1.0
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, bbox_target, cls_target

def create_dataloaders(image_dir, label_dir, batch_size=32, num_workers=4):
    # Create dataset
    dataset = FLIRDataset(image_dir, label_dir)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

if __name__ == "__main__":
    # Test the dataset
    image_dir = "data/thermal/images/train_flir_images"
    label_dir = "data/thermal/labels/train_flir_images"
    
    dataloader = create_dataloaders(image_dir, label_dir, batch_size=2)
    
    # Get a batch
    for images, bboxes, classes in dataloader:
        print(f"Image batch shape: {images.shape}")
        print(f"Bounding box batch shape: {bboxes.shape}")
        print(f"Class batch shape: {classes.shape}")
        break 