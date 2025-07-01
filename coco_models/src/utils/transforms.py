import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import Dict, List, Union, Tuple
import torch.nn.functional as F_nn
import math


import logging

log = logging.getLogger(__name__)

class GPUCollate:
    def __init__(self, device, transforms):
        self.device = device
        self.transforms = transforms.to(device) #moving transforms to GPU 
        
    def __call__(self, batch):
        """
        - images: Tensor[batch_size, C, H, W]
        - targets: List[Dict] where each dict has:
            - boxes: Tensor[N, 4]
            - labels: Tensor[N] (all 1s for person class)
            - image_id: Tensor[1] (single tensor for each image)
        """
        images = []
        targets = []
    
        #for image, target in batch:
        for i, (image, target) in enumerate(batch): #see if i can delete this 
            images.append(image)
            # Create target dict with single tensors (not lists)
            target_dict = {
                'boxes': target['boxes'],  # already a tensor from dataset
                'labels': target['labels'],  # preserve labels from dataset
                'image_id': target['image_id']  # already a tensor from dataset
            }
            targets.append(target_dict)
    
        # Stack images into a single tensor (they should all be the same size) and moving to device
        images = torch.stack(images, dim=0).to(self.device)

        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets] #moving targets to device
        
        # Apply transforms to images and targets
        images, targets = self.transforms(images, targets)
        
        return images, targets

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
                self.transforms.append(('flip_h', t['params']['p']))
            elif t['name'] == 'RandomVerticalFlip':
                self.transforms.append(('flip_v', t['params']['p']))
            elif t['name'] == 'RandomRotation':
                self.transforms.append(('rotate', t['params']['degrees']))
    
    def to(self, device):
        self.device = device
        return self

    
    def __call__(self, images, targets):
        #transforms to both image and bounding boxes
        #images are already tensors when using GPUCollate
        assert images.shape[1] == 1, "Expected single channel images"
        assert len(images.shape) == 4, f"expected batched images [B,C,H,W], got shape {images.shape}"
        
        for t_name, *params in self.transforms:

            #images are batched tensors of [batch_size, C, H, W]
            if t_name == 'resize':
                size = params[0]
                # resize with padding
                images, resize_info = resize_with_padding(images, size)

                target_h, target_w = size
                
                # Transform boxes 
                for i in range(len(targets)):
                    if targets[i]is not None and 'boxes' in targets[i]:
                        boxes = targets[i]['boxes']
                        boxes = transform_boxes(
                            boxes,
                            resize_info['scale'],
                            resize_info['pad_left'],
                            resize_info['pad_top'],
                            target_w,
                            target_h
                        )
                        targets[i]['boxes'] = boxes

            elif t_name == 'normalize':
                mean, std = params
                images = F.normalize(images, mean=mean, std=std)
            
            elif t_name == 'flip_h':
                p = params[0]
                flip_mask = (torch.rand(images.shape[0], device=images.device) < p) #generating mask of booleans for whethr to flip or not
                # Flip whole batch where mask is True
                images[flip_mask] = F.hflip(images[flip_mask])
                for i, flip_true in enumerate(flip_mask):
                    if flip_true and 'boxes' in targets[i]:
                        boxes = targets[i]['boxes']
                        boxes[:, [0, 2]] = images.shape[-1] - boxes[:, [2, 0]]
                        targets[i]['boxes'] = boxes

            elif t_name == 'flip_v':
                p = params[0]
                flip_mask = (torch.rand(images.shape[0], device=images.device) < p) 
                images[flip_mask] = F.vflip(images[flip_mask])
                for i, flip_true in enumerate(flip_mask):
                    if flip_true and 'boxes' in targets[i]:
                        boxes = targets[i]['boxes']
                        # new_y = height - old_y 
                        boxes[:, [1, 3]] = images.shape[-2] - boxes[:, [3, 1]]
                        targets[i]['boxes'] = boxes
            
            elif t_name == 'rotate':
                degrees = params[0]
                p = 0.3  # probability of applying rotation, could be made configurable
                rotate_mask = (torch.rand(images.shape[0], device=images.device) < p)
                # generate angles for images that will be rotated
                angles = torch.zeros(images.shape[0], device=images.device)
                angles[rotate_mask] = torch.empty(rotate_mask.sum(), device=images.device).uniform_(-degrees, degrees)
                images, targets = rotate_batch(images, targets, angles)
        
        return images, targets
    
def build_transforms(cfg: Dict, is_train: bool = True, test: bool = False) -> DetectionTransform:
    if is_train:
        transforms = cfg.dataset.transform.train
    elif test:
        transforms = cfg.dataset.transform.test
    else:
        transforms = cfg.dataset.transform.val
    
    return DetectionTransform(transforms) 

def resize_with_padding(image: torch.Tensor, target_size: Tuple[int, int]) -> Tuple[torch.Tensor, Dict[str, float]]:
   #maintain aspect ratio and add padding
    b, c, h, w = image.shape
    target_h, target_w = target_size
    
    # calc scaling factors (to find limiting facotr )
    scale_h = target_h / h
    scale_w = target_w / w
    scale = min(scale_h, scale_w)
    
    # new size
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize image
    resized_image = F_nn.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False) #using interpolate since they are already tensors (not PIL)
    
    # padding
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Add padding
    padded_image = F_nn.pad(resized_image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    return padded_image, {
        'scale': scale,
        'pad_left': pad_left,
        'pad_top': pad_top #resize info
    }

def transform_boxes(boxes: torch.Tensor, scale: float, pad_left: int, pad_top: int, target_w: int, target_h: int) -> torch.Tensor:
    #bounding boxes tranformed in line wiht image tranforms 
    #chnaged (adding clamp to ensure boxes are within image bounds)
    transformed_boxes = boxes.clone()
    transformed_boxes[:, [0, 2]] = (transformed_boxes[:, [0, 2]] * scale + pad_left).clamp(0, max=target_w) # x1 and x2 (since already converted in flir_dataset.py)
    transformed_boxes[:, [1, 3]] = (transformed_boxes[:, [1, 3]] * scale + pad_top).clamp(0, max=target_h)   # y1 and y2
    
    return transformed_boxes

def rotate_image_and_boxes(image: torch.Tensor, boxes: torch.Tensor, angle: float, expand: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    #rotation to images and bounding boxes
    # image dims
    c, h, w = image.shape
    
    # Convert to PIL =
    image_pil = F.to_pil_image(image)
    
    # Rotate image
    rotated_image = F.rotate(image_pil, angle, expand=expand)
    rotated_image = F.to_tensor(rotated_image)
    
    if boxes is None or len(boxes) == 0:
        return rotated_image, boxes
    
    # Convert boxes to points 
    points = boxes_to_points(boxes)
    
    # Rotate points
    center = (w / 2, h / 2)
    rotated_points = rotate_points(points, center, -angle)  # PIL rotation is anticlockwise -- neg angle 
    
    rotated_boxes = points_to_boxes(rotated_points)

    # checks to ensure boxes are valid (CHANGED)
    if rotated_boxes.shape[0] > 0:
        x1, y1, x2, y2 = rotated_boxes.unbind(-1)
        valid = (x2 > x1) & (y2 > y1)
        rotated_boxes = rotated_boxes[valid] #filtering out any invalid boxes
    
    return rotated_image, rotated_boxes

def boxes_to_points(boxes: torch.Tensor) -> torch.Tensor:
    #Convert boxes to 4 corner points
    x1, y1, x2, y2 = boxes.unbind(-1)
    points = torch.stack([
        torch.stack([x1, y1], dim=1),  # top left
        torch.stack([x2, y1], dim=1),  # top right
        torch.stack([x2, y2], dim=1),  # bottom right
        torch.stack([x1, y2], dim=1),  # bottom left
    ], dim=1)
    return points

def points_to_boxes(points: torch.Tensor) -> torch.Tensor:
    #Convert corner points to boxes
    min_coords, _ = torch.min(points, dim=1)
    max_coords, _ = torch.max(points, dim=1)
    return torch.cat([min_coords, max_coords], dim=1)

def rotate_points(points: torch.Tensor, center: Tuple[float, float], rotMatrix: torch.Tensor) -> torch.Tensor:
    #Rotate points around center by angle (in degrees)
    cx, cy = center
    points = points.clone()
    
    # Translate to origin
    points[..., 0] -= cx
    points[..., 1] -= cy
    
    x_prime = points[..., 0] * rotMatrix[0,0] + points[..., 1] * rotMatrix[1,0] #rotation matrix multiplication 
    y_prime = points[..., 0] * rotMatrix[0,1] + points[..., 1] * rotMatrix[1,1]
    
    # Translate back
    points[..., 0] = x_prime + cx
    points[..., 1] = y_prime + cy
    
    return points

def rotate_batch(images,targets,angles, expand=False):
    # images: [B, C, H, W]
    # angles: [B] tensor of different angles for each image
    angles_rad = (angles * math.pi / 180)
    # Batch rotation matrix
    theta = torch.zeros(images.shape[0], 2, 3, device=images.device) #2x3 matrix required for affine_grid but final column remains empty as no translation is applied 
    theta[:, 0, 0] = torch.cos(angles_rad)
    theta[:, 0, 1] = -torch.sin(angles_rad)
    theta[:, 1, 0] = torch.sin(angles_rad)
    theta[:, 1, 1] = torch.cos(angles_rad)

    grid = F_nn.affine_grid(theta, images.shape, align_corners=False)
    rotated_images = F_nn.grid_sample(images, grid, align_corners=False)
    
    center = (images.shape[-1] / 2, images.shape[-2] / 2)
    for i, angle in enumerate(angles):
        if targets[i] is not None and 'boxes' in targets[i]:
            boxes = targets[i]['boxes']
            # Use your existing functions
            points = boxes_to_points(boxes)
            rotated_points = rotate_points(points, center, theta[i])  # Negative angle as per your current code
            targets[i]['boxes'] = points_to_boxes(rotated_points)
    
    return rotated_images, targets