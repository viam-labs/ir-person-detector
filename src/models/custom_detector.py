#set up to work with the original flir dataset in COCO format
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ThermalDetector(nn.Module):
    def __init__(self):
        super(ThermalDetector, self).__init__()
        
        # CNN backbone
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(), #applying relu activation
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Detection head
        self.detector = nn.Sequential(
            nn.Linear(128 * 80 * 64, 512),  # Adjust based on your input size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 5)  # 4 for bbox (x, y, w, h) + 1 for class
        )
        
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Flatten
        features = features.view(features.size(0), -1)
        
        # Get predictions
        predictions = self.detector(features)
        
        # Split into bbox and class predictions
        bbox_pred = predictions[:, :4]
        cls_pred = predictions[:, 4:]
        
        return bbox_pred, cls_pred

class FLIRDataset(Dataset):
    def __init__(self, json_file, thermal_dir, transform=None):
        """
        Args:
            json_file (string): Path to the COCO format annotation file
            thermal_dir (string): Directory with thermal images
            transform (callable, optional): Optional transform to be applied on images
        """
        with open(json_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.thermal_dir = thermal_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((640, 512)),  # Original FLIR resolution
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # For thermal images
        ])
        
        # Create image id to annotations mapping
        self.img_to_anns = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Get all image ids
        self.image_ids = list(self.img_to_anns.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Get image id
        img_id = self.image_ids[idx]
        
        # Find image info
        img_info = next(img for img in self.annotations['images'] if img['id'] == img_id)
        
        # Load thermal image
        img_path = os.path.join(self.thermal_dir, img_info['file_name'])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Get annotations for this image
        anns = self.img_to_anns[img_id]
        
        # Initialize targets
        bbox_target = torch.zeros(4, dtype=torch.float32)  # [x, y, w, h]
        cls_target = torch.zeros(1, dtype=torch.float32)   # 1 for person
        
        for ann in anns:
            if ann['category_id'] == 1:  # Assuming 1 is person
                bbox = ann['bbox']  # [x, y, w, h]
                bbox_target = torch.tensor(bbox, dtype=torch.float32)
                cls_target[0] = 1.0
                break
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, bbox_target, cls_target

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    print(f"\nStarting training on {device}")
    print(f"Total epochs: {num_epochs}")
    print(f"Total batches per epoch: {len(train_loader)}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Total training samples: {len(train_loader.dataset)}\n")
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_bbox_loss = 0.0
        running_cls_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("=" * 50)
        
        for batch_idx, (images, bboxes, classes) in enumerate(train_loader):
            # Move data to device
            images = images.to(device)
            bboxes = bboxes.to(device)
            classes = classes.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            bbox_pred, cls_pred = model(images)
            
            # Calculate loss
            bbox_loss = criterion(bbox_pred, bboxes)
            cls_loss = criterion(cls_pred, classes)
            loss = bbox_loss + cls_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running losses
            running_loss += loss.item()
            running_bbox_loss += bbox_loss.item()
            running_cls_loss += cls_loss.item()
            
            # Print batch progress
            if (batch_idx + 1) % 10 == 0:  # Print every 10 batches
                print(f"Batch {batch_idx + 1}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f} "
                      f"(Bbox: {bbox_loss.item():.4f}, Cls: {cls_loss.item():.4f})")
        
        # Print epoch summary
        epoch_loss = running_loss / len(train_loader)
        epoch_bbox_loss = running_bbox_loss / len(train_loader)
        epoch_cls_loss = running_cls_loss / len(train_loader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Average Loss: {epoch_loss:.4f}")
        print(f"Average Bbox Loss: {epoch_bbox_loss:.4f}")
        print(f"Average Classification Loss: {epoch_cls_loss:.4f}")
        print("=" * 50)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = ThermalDetector().to(device)
    print("Model created and moved to device")
    
    # Create dataset and dataloader
    dataset = FLIRDataset(
        json_file='/Users/isha.yerramilli-rao/FLIR_ADAS_v2/images_thermal_train/coco.json',
        thermal_dir='/Users/isha.yerramilli-rao/FLIR_ADAS_v2/images_thermal_train'
    )
    print(f"Dataset loaded with {len(dataset)} images")
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"DataLoader created with batch size {dataloader.batch_size}")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # For bbox regression
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("\nLoss function and optimizer initialized")
    
    # Train model
    print("\nStarting training...")
    train_model(model, dataloader, criterion, optimizer, device) 