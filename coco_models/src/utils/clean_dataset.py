import json
from pathlib import Path
import logging
from tqdm import tqdm
from PIL import Image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def is_valid_bbox(bbox, img_width, img_height):
    """Check if bbox is valid.
    bbox format: [x, y, width, height]
    """
    x, y, w, h = bbox
    
    # Check for zero or negative dimensions
    if w <= 0 or h <= 0:
        return False
    
    # Check if box extends beyond image boundaries
    if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
        return False
    
    return True

def clean_dataset(json_path, img_dir, output_json_path):
    """Clean COCO dataset by removing invalid annotations and their corresponding images if no valid annotations remain."""
    
    # Load JSON
    log.info(f"Loading annotations from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Initialize counters
    total_images = len(data['images'])
    total_annotations = len(data['annotations'])
    removed_annotations = 0
    removed_images = 0
    
    # Create mapping of image_id to image info for faster lookup
    image_info = {img['id']: img for img in data['images']}
    
    # Create mapping of image_id to list of annotation indices
    image_to_anns = {}
    for idx, ann in enumerate(data['annotations']):
        img_id = ann['image_id']
        if img_id not in image_to_anns:
            image_to_anns[img_id] = []
        image_to_anns[img_id].append(idx)
    
    # Track valid annotations and images
    valid_annotations = []
    valid_image_ids = set()
    
    # Process annotations
    log.info("Checking annotations...")
    for img_id, ann_indices in tqdm(image_to_anns.items()):
        if img_id not in image_info:
            continue
            
        # Get image dimensions
        img_path = Path(img_dir) / image_info[img_id]['file_name']
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            log.warning(f"Could not open image {img_path}: {e}")
            continue
        
        # Check each annotation for this image
        valid_anns_for_image = []
        for idx in ann_indices:
            ann = data['annotations'][idx]
            if is_valid_bbox(ann['bbox'], img_width, img_height):
                valid_anns_for_image.append(ann)
            else:
                removed_annotations += 1
                log.debug(f"Removed invalid bbox {ann['bbox']} from image {img_id}")
        
        # If image has valid annotations, keep it
        if valid_anns_for_image:
            valid_annotations.extend(valid_anns_for_image)
            valid_image_ids.add(img_id)
        else:
            removed_images += 1
            log.debug(f"Removed image {img_id} - no valid annotations")
    
    # Create new dataset with only valid images and annotations
    cleaned_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': data['categories'],
        'images': [img for img in data['images'] if img['id'] in valid_image_ids],
        'annotations': valid_annotations
    }
    
    # Save cleaned dataset
    log.info(f"Saving cleaned dataset to {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    # Log statistics
    log.info(f"Original images: {total_images}")
    log.info(f"Original annotations: {total_annotations}")
    log.info(f"Removed images: {removed_images}")
    log.info(f"Removed annotations: {removed_annotations}")
    log.info(f"Remaining images: {len(cleaned_data['images'])}")
    log.info(f"Remaining annotations: {len(cleaned_data['annotations'])}")

def shift_category_ids(json_path):
    with open(json_path) as f:
        data = json.load(f)
    for ann in data["annotations"]:
        ann["category_id"] += 1
    for cat in data["categories"]:
        cat["id"] += 1

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Clean COCO dataset by removing invalid annotations')
    parser.add_argument('--json_path', type=str, required=True, help='Path to input COCO JSON file')
    parser.add_argument('--img_dir', type=str, required=True, help='Path to directory containing images')
    #parser.add_argument('--output_json', type=str, required=True, help='Path to save cleaned JSON file')
    
    args = parser.parse_args()
    shift_category_ids(args.json_path)
    clean_dataset(args.json_path, args.img_dir, args.output_json) 
    
