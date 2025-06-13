from pathlib import Path
import json
from PIL import Image
import sys
from tqdm import tqdm

def check_image_sizes(json_file, thermal_dir):
    # Load annotations
    print(f"\nChecking dataset: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Keep track of different sizes
    size_counts = {}
    mismatched_files = []
    missing_files = []
    expected_size = (640, 640)
    total_images = len(data['images'])
    
    # Check each image
    print(f"Found {total_images} images in annotations")
    for img_info in tqdm(data['images']):
        img_path = Path(thermal_dir) / img_info['file_name']
        
        # Also check the size recorded in COCO annotations
        coco_width = img_info.get('width')
        coco_height = img_info.get('height')
        if (coco_width, coco_height) != expected_size:
            print(f"\nWarning: COCO annotation size mismatch for {img_info['file_name']}")
            print(f"Annotation reports size: ({coco_width}, {coco_height})")
        
        try:
            with Image.open(img_path) as img:
                size = img.size
                if size != expected_size:
                    mismatched_files.append((img_path, size, (coco_width, coco_height)))
                size_counts[size] = size_counts.get(size, 0) + 1
        except FileNotFoundError:
            missing_files.append(img_path)
        except Exception as e:
            print(f"\nError reading {img_path}: {e}")
    
    # Print results
    print("\nResults:")
    print("-" * 50)
    print(f"Total images in annotations: {total_images}")
    print(f"Successfully read images: {sum(size_counts.values())}")
    print(f"Missing files: {len(missing_files)}")
    
    print("\nImage size distribution:")
    for size, count in size_counts.items():
        print(f"{size}: {count} images ({count/total_images*100:.1f}%)")
    
    if mismatched_files:
        print("\nFiles with unexpected dimensions:")
        for path, size in mismatched_files:
            print(f"{path}: {size}")
    else:
        print("\nAll images are 640x640 as expected.")

if __name__ == "__main__":
    # Check training set
    check_image_sizes(
        "/Users/isha.yerramilli-rao/FLIR_ADAS_v2/images_thermal_train/coco.json",
        "/Users/isha.yerramilli-rao/FLIR_ADAS_v2/images_thermal_train"
    )
    
    # Check validation set
    check_image_sizes(
        "/Users/isha.yerramilli-rao/FLIR_ADAS_v2/images_thermal_val/coco.json",
        "/Users/isha.yerramilli-rao/FLIR_ADAS_v2/images_thermal_val"
    )
    
    # Check test set
    check_image_sizes(
        "/Users/isha.yerramilli-rao/FLIR_ADAS_v2/images_thermal_test/coco.json",
        "/Users/isha.yerramilli-rao/FLIR_ADAS_v2/images_thermal_test"
    ) 