import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image

class FLIRDataset(Dataset):
    def __init__(self, json_file, thermal_dir, transform=None):
        self.thermal_dir = Path(thermal_dir)
        self.transform = transform
        
        # Load annotations
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Create image_id to annotations mapping
        self.annotations = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        # Keep only images that have annotations
        self.images = [img for img in data['images'] if img['id'] in self.annotations]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_info = self.images[idx]
        img_path = self.thermal_dir / img_info['file_name']
        image = Image.open(img_path)
        
        # Get annotations for this image
        img_anns = self.annotations[img_info['id']]
        
        # Extract bounding boxes
        boxes = []
        for ann in img_anns:
            x, y, w, h = ann['bbox']
            # Convert from [x,y,w,h] to [x1,y1,x2,y2] format
            boxes.append([x, y, x+w, y+h])
        
        # Convert to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'image_id': torch.tensor([img_info['id']])
        }
        
        # Apply transforms if any
        if self.transform:
            image, target = self.transform(image, target)
        
        return image, target 