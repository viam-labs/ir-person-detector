import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import Dict, List, Union, Tuple

def custom_collate_fn(batch):
    """batches images and targets into a single dictionary.
    
    Args:
        batch: List of tuples (image, target)
        
    Returns:
        images: Tensor of shape [batch_size, C, H, W]
        targets: Dictionary containing:
            - boxes: List of tensors, each of shape [num_boxes, 4]
            - image_id: List of tensors, each of shape [1]
    """
    images = []
    boxes = []
    image_ids = []
    
    for image, target in batch:
        images.append(image)
        boxes.append(target['boxes'])
        image_ids.append(target['image_id'])
    
    # Stack images (they should all be the same size)
    images = torch.stack(images, dim=0)
    
    # Keep boxes and image_ids as lists since they can have different sizes
    return images, {
        'boxes': boxes,
        'image_id': image_ids
    }

class DetectionTransform:
    # transform that can be applied to images and bounding boxes
    def __init__(self, transforms: List[Dict]):
        self.transforms = []
        for t in transforms:
            if t['name'] == 'Normalize':
                self.transforms.append(('normalize', t['params']['mean'], t['params']['std']))
            elif t['name'] == 'RandomHorizontalFlip':
                self.transforms.append(('flip', t['params']['p']))
            elif t['name'] == 'ColorJitter':
                self.transforms.append(('color', t['params']))
    
    def __call__(self, image, target):
        """Apply transforms to both image and bounding boxes."""
        # Convert PIL Image to tensor
        image = F.to_tensor(image)
        
        for t_name, *params in self.transforms:
            if t_name == 'normalize':
                mean, std = params
                image = F.normalize(image, mean=mean, std=std)
            
            elif t_name == 'flip':
                p = params[0]
                if torch.rand(1) < p:
                    image = F.hflip(image)
                    if target is not None and 'boxes' in target:
                        boxes = target['boxes']
                        # Flip boxes: new_x = width - old_x
                        boxes[:, [0, 2]] = image.shape[-1] - boxes[:, [2, 0]]
                        target['boxes'] = boxes
            
            elif t_name == 'color':
                params = params[0]
                image = T.ColorJitter(**params)(image)
        
        return image, target

def build_transforms(cfg: Dict, is_train: bool = True) -> DetectionTransform:
    """Build transforms from config."""
    if is_train:
        transforms = cfg.dataset.transform.train
    else:
        transforms = cfg.dataset.transform.val
    return DetectionTransform(transforms) 