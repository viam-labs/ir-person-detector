import os
import json
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def resize_and_pad(image, target_size):
    """Resize image to target size while maintaining aspect ratio and pad if necessary."""
    # Get current and desired ratio
    current_ratio = image.size[0] / image.size[1]
    target_ratio = target_size[0] / target_size[1]
    
    # Calculate new size maintaining aspect ratio
    if current_ratio > target_ratio:
        # Width is limiting factor
        new_width = target_size[0]
        new_height = int(new_width / current_ratio)
    else:
        # Height is limiting factor
        new_height = target_size[1]
        new_width = int(new_height * current_ratio)
    
    # Resize image
    image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    
    # Create new image with padding
    new_image = Image.new('L', target_size, 0)  # 'L' for grayscale, black padding
    
    # Calculate padding
    left_pad = (target_size[0] - new_width) // 2
    top_pad = (target_size[1] - new_height) // 2
    
    # Paste resized image onto padded background
    new_image.paste(image, (left_pad, top_pad))
    
    return new_image, (left_pad, top_pad, new_width, new_height)

def update_annotations(json_path, padding_info):
    """Update COCO annotations to account for padding and resizing."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Update image dimensions in annotations
    for img in data['images']:
        img['width'] = 640
        img['height'] = 640
        
        # Get padding info for this image
        if img['file_name'] in padding_info:
            left_pad, top_pad, new_width, new_height = padding_info[img['file_name']]
            
            # Find annotations for this image
            for ann in data['annotations']:
                if ann['image_id'] == img['id']:
                    # Update bbox coordinates
                    x, y, w, h = ann['bbox']
                    
                    # Scale coordinates based on resize
                    scale_x = new_width / img['orig_width']
                    scale_y = new_height / img['orig_height']
                    
                    x = x * scale_x + left_pad
                    y = y * scale_y + top_pad
                    w = w * scale_x
                    h = h * scale_y
                    
                    ann['bbox'] = [x, y, w, h]
    
    # Save updated annotations
    with open(json_path, 'w') as f:
        json.dump(data, f)

def process_dataset(base_dir):
    """Process all images in a dataset split."""
    target_size = (640, 640)
    padding_info = {}
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        # Setup paths
        img_dir = Path(base_dir) / f"images_thermal_{split}"
        json_path = img_dir / "coco.json"
        
        # Load annotations to get file list
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Store original dimensions
        for img in data['images']:
            with Image.open(img_dir / img['file_name']) as im:
                img['orig_width'], img['orig_height'] = im.size
        
        # Process each image
        for img_info in tqdm(data['images']):
            img_path = img_dir / img_info['file_name']
            
            try:
                # Open and resize image
                with Image.open(img_path) as img:
                    resized_img, pad_info = resize_and_pad(img, target_size)
                    padding_info[img_info['file_name']] = pad_info
                    
                    # Save back to same location
                    resized_img.save(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Update annotations
        update_annotations(json_path, padding_info)

if __name__ == "__main__":
    base_dir = "/Users/isha.yerramilli-rao/FLIR_ADAS_v2"
    process_dataset(base_dir)