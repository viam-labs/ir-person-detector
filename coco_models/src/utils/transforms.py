import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import Dict, List, Union, Tuple

def custom_collate_fn(batch): #called in train.py

    #  returns targets: dictionary containing
    #       - boxes: List of tensors, each of shape [num_boxes, 4]
    #      - image_id: List of tensors, each of shape [1]
    #ensures proper batching of detection data to be used in dataloader 

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
            if t['name'] == 'Resize':
                self.transforms.append(('resize', t['params']['size']))
            elif t['name'] == 'Normalize':
                self.transforms.append(('normalize', t['params']['mean'], t['params']['std']))
            elif t['name'] == 'RandomHorizontalFlip':
                self.transforms.append(('flip', t['params']['p']))
            elif t['name'] == 'ColorJitter':
                self.transforms.append(('color', t['params']))
    
    def __call__(self, image, target):
        #transforms to both image and bounding boxes
        image = F.to_tensor(image) 
        
        # checking image has 1 channel
        if image.shape[0] != 1:
            raise ValueError(f"Expected 1 channel (thermal), got {image.shape[0]} channels")
        
        for t_name, *params in self.transforms:
            if t_name == 'resize':
                size = params[0]
                # resize with padding
                image, resize_info = self.resize_with_padding(image, size)
                
                # Transform boxes 
                if target is not None and 'boxes' in target:
                    boxes = target['boxes']
                    boxes = self.transform_boxes(
                        boxes,
                        resize_info['scale'],
                        resize_info['pad_left'],
                        resize_info['pad_top']
                    )
                    target['boxes'] = boxes

            elif t_name == 'normalize':
                mean, std = params
                image = F.normalize(image, mean=mean, std=std)
            
            elif t_name == 'flip':
                p = params[0]
                if torch.rand(1) < p:
                    image = F.hflip(image)
                    if target is not None and 'boxes' in target:
                        boxes = target['boxes']
                        # new_x = width - old_x (flip)
                        boxes[:, [0, 2]] = image.shape[-1] - boxes[:, [2, 0]]
                        target['boxes'] = boxes
        return image, target

def build_transforms(cfg: Dict, is_train: bool = True) -> DetectionTransform:
    if is_train:
        transforms = cfg.dataset.transform.train
    else:
        transforms = cfg.dataset.transform.val
    return DetectionTransform(transforms) 

def resize_with_padding(self, image: torch.Tensor, target_size: Tuple[int, int]) -> Tuple[torch.Tensor, Dict[str, float]]:
   #maintain aspect ratio and add padding
    c, h, w = image.shape
    target_h, target_w = target_size
    
    # calc scaling factors (to find limiting facotr )
    scale_h = target_h / h
    scale_w = target_w / w
    scale = min(scale_h, scale_w)
    
    # new size
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize image
    resized_image = F.resize(image, [new_h, new_w])
    
    # padding
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Add padding
    padded_image = F.pad(resized_image, [pad_left, pad_top, pad_right, pad_bottom], value=0)
    
    return padded_image, {
        'scale': scale,
        'pad_left': pad_left,
        'pad_top': pad_top #resize info
    }

def transform_boxes(self, boxes: torch.Tensor, scale: float, pad_left: int, pad_top: int) -> torch.Tensor:
    #bounding boxes tranformed in line wiht image tranforms 
    # boxes are [x1, y1, w, h]
    transformed_boxes = boxes.clone()
    
    # Scale coordinates
    transformed_boxes[:, 0] = boxes[:, 0] * scale + pad_left  # x1
    transformed_boxes[:, 1] = boxes[:, 1] * scale + pad_top   # y1
    transformed_boxes[:, 2] = boxes[:, 2] * scale             # w
    transformed_boxes[:, 3] = boxes[:, 3] * scale             # h
    
    return transformed_boxes