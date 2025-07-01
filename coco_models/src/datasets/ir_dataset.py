import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
import torch.nn as nn
import logging
import torchvision.transforms.functional as F
log = logging.getLogger(__name__)
class IRDataset(Dataset):
    def __init__(self, json_file, thermal_dir):
        self.thermal_dir = Path(thermal_dir)
        log.info(f"thermal_dir: {self.thermal_dir}")
        
        # Load annotations
        with open(json_file, 'r') as f:
            data = json.load(f)
                
        self.images = data['images']
        self.annotations = data['annotations']
        # Create image_id to annotations mapping
        self.annotations = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_info = self.images[idx]
        img_path = self.thermal_dir / img_info['file_name']
        if not img_path.exists():
            raise RuntimeError(f"TRYING TO OPEN NON-EXISTENT FILE: {img_path}")
        image = Image.open(img_path)
        # Convert to tensor and ensure single channel
        image = F.to_tensor(image)

        image = image[0:1]  #filtering to one channel only
        # Get annotations for this image
        img_anns = self.annotations[img_info['id']]
        
        # Extract bounding boxes
        boxes = []
        for ann in img_anns:
            x, y, w, h = ann['bbox']
            #Convert from [x,y,w,h] to [x1,y1,x2,y2] format
            boxes.append([x, y, x+w, y+h])
        
        # Convert to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': torch.zeros(boxes.shape[0], dtype=torch.int64),  # all 0s for person class
            'image_id': torch.tensor([img_info['id']])
        }
        
        # Apply transforms if any
        #if self.transform:
        #    image, target = self.transform(image, target) #not applying transforms here to instead apply them in the dataloader with GPU
        
        return image, target