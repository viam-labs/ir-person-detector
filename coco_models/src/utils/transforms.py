import torch
import torchvision.transforms as T
from typing import Dict, List, Union, Tuple
import torch.nn.functional as F

def custom_collate_fn(batch):
    """Custom collate function to handle batches with different numbers of bounding boxes.
    
    Args:
        batch: List of tuples (image, target)
        
    Returns:
        images: Tensor of shape [batch_size, C, H, W]
        targets: List of targets (each target can have different number of boxes)
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # Stack images (they should all be the same size)
    images = torch.stack(images, dim=0)
    
    return images, targets

class DetectionTransform:
    # transform that can be applied to images and bounding boxes
    def __init__(self, transforms: List[Dict]):
        self.transforms = []
        for t in transforms:
            if t['name'] == 'Normalize':
                self.transforms.append(('normalize', T.Normalize(
                    mean=t['params']['mean'],
                    std=t['params']['std']
                )))
            elif t['name'] == 'RandomHorizontalFlip':
                self.transforms.append(('flip', T.RandomHorizontalFlip(t['params']['p'])))
            elif t['name'] == 'ColorJitter':
                self.transforms.append(('color', T.ColorJitter(
                    brightness=t['params'].get('brightness', 0),
                    contrast=t['params'].get('contrast', 0)
                )))
    
    def __call__(self, image: torch.Tensor, target: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Apply transforms to both image and target
        Args:
            image: Tensor of shape [C, H, W]
            target: Dict containing 'boxes' and 'labels'
        Returns:
            Transformed image and target
        """
        boxes = target['boxes']  # shape: [N, 4] where N is number of boxes
        
        for transform_type, t in self.transforms:
            if transform_type == 'flip' and isinstance(t, T.RandomHorizontalFlip):
                if torch.rand(1) < t.p:  # flip with probability p to maintain randomization
                    image = torch.flip(image, dims=[-1])
                    
                    # new_x = W - old_x - width
                    if len(boxes) > 0:  # Only process if there are boxes
                        boxes_x1 = boxes[:, 0].clone()
                        boxes_x2 = boxes[:, 2].clone()
                        boxes[:, 0] = image.shape[-1] - boxes_x2
                        boxes[:, 2] = image.shape[-1] - boxes_x1
            
            elif transform_type in ['color', 'normalize']:
                # transforms only affect the image
                image = t(image)
        
        target['boxes'] = boxes
        return image, target

def build_transforms(cfg: Dict, is_train: bool = True) -> DetectionTransform:
    # accessing transforms from config  
    transforms = cfg.dataset.transform.train if is_train else cfg.dataset.transform.val
    return DetectionTransform(transforms) 