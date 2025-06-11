import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

class FLIRDataset(Dataset):
    def __init__(self, json_file: Path, thermal_dir: Path, transform=None):
        """
        Args:
            json_file (Path): Path to the COCO format annotation file
            thermal_dir (Path): Directory with thermal images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.thermal_dir = thermal_dir
        self.transform = transform
        
        # Load annotations
        with open(json_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create image id to annotations mapping
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Create image id to image info mapping
        self.img_to_info = {img['id']: img for img in self.coco_data['images']}
        
        # Get list of image ids
        self.ids = list(self.img_to_info.keys())
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.img_to_info[img_id]
        anns = self.img_to_anns.get(img_id, [])
        
        # Load image
        img_path = self.thermal_dir / img_info['file_name']
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert to tensor and normalize
        img = torch.from_numpy(img).float() / 255.0
        img = img.unsqueeze(0)  # Add channel dimension
        
        # Prepare targets
        boxes = []
        labels = []
        
        for ann in anns:
            # Get bounding box
            bbox = ann['bbox']  # [x, y, width, height]
            boxes.append(bbox)
            
            # Get category id (0 for person)
            labels.append(0)  # Assuming person is the only class
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }
        
        if self.transform:
            img, target = self.transform(img, target)
        
        return img, target 