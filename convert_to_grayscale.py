import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from functools import partial
import time

def convert_to_grayscale(img_path, output_path):
    """Convert an image to true grayscale"""
    try:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            return False
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Save as grayscale
        cv2.imwrite(str(output_path), gray)
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return False

def process_dataset(input_dir, output_dir, num_workers=None):
    """Process all images in the input directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
    total_images = len(image_files)
    
    print(f"Found {total_images} images to process")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, int(mp.cpu_count() * 0.75))
    
    print(f"Using {num_workers} workers for processing")
    
    # Create a partial function with fixed arguments
    process_func = partial(convert_to_grayscale, output_path=output_path)
    
    # Start timing
    start_time = time.time()
    
    # Process images using multiprocessing
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, image_files),
            total=total_images,
            desc="Converting to grayscale"
        ))
    
    # Calculate statistics
    successful = sum(results)
    failed = total_images - successful
    elapsed_time = time.time() - start_time
    
    print(f"\nProcessing completed:")
    print(f"Total images: {total_images}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per image: {elapsed_time/total_images:.2f} seconds")

if __name__ == "__main__":
    # Example usage
    input_directory = "/Users/isha.yerramilli-rao/data/thermal/images"
    output_directory = "/Users/isha.yerramilli-rao/data/thermal/images_grayscale"
    
    process_dataset(input_directory, output_directory) 