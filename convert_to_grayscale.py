#redundant code since each image is alrayd only 1 channel!11
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
from pathlib import Path
import json
import shutil

class IRDataset(Dataset):
    def __init__(self, image_dir, transform=None, save_dir=None, format='yolo', test_mode=False):
        self.image_dir = image_dir
        self.transform = transform
        self.save_dir = save_dir
        self.format = format.lower()
        self.test_mode = test_mode
        
        if save_dir:
            # Create format-specific directories
            if self.format == 'yolo':
                os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
                os.makedirs(os.path.join(save_dir, 'labels'), exist_ok=True)
            elif self.format == 'coco':
                os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
                os.makedirs(os.path.join(save_dir, 'annotations'), exist_ok=True)
        
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if test_mode:
            self.image_files = self.image_files[:1]  # Take only the first image for testing
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Convert to single channel using tensor operations
        if image.shape[0] == 3:  # from RGB
            # Verify all channels are equal (they should be triplicated)
            assert torch.all(torch.eq(image[0], image[1])) and torch.all(torch.eq(image[1], image[2])), \
                "Channels are not equal in the input image"
            # first channel taken since all channels are identical 
            ir_image = image[0:1]  # output shape: [1, H, W]
            
            # Save the converted image if save_dir is specified
            if self.save_dir:
                # Convert tensor to PIL Image
                ir_pil = transforms.ToPILImage()(ir_image)
                
                if self.format == 'yolo':
                    # Save image in YOLO format structure
                    save_path = os.path.join(self.save_dir, 'images', self.image_files[idx])
                    ir_pil.save(save_path)
                    if self.test_mode:
                        print(f"Saved test image to: {save_path}")
                    
                elif self.format == 'coco':
                    # Save image in COCO format structure
                    save_path = os.path.join(self.save_dir, 'images', self.image_files[idx])
                    ir_pil.save(save_path)
                    if self.test_mode:
                        print(f"Saved test image to: {save_path}")
            
            return ir_image
        return image

def get_ir_dataloader(image_dir, batch_size=16, num_workers=4, img_size=640, save_dir=None, format='yolo', test_mode=False):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), #creating 640x640 square images
        transforms.ToTensor(), #torch tensor
    ])
    
    dataset = IRDataset(image_dir, transform=transform, save_dir=save_dir, format=format, test_mode=test_mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    
    return dataloader

def copy_annotations(src_dir, dst_dir, format='yolo', test_mode=False):
    """Copy annotation files to the appropriate format structure"""
    if format == 'yolo':
        # Copy YOLO format labels
        label_dir = os.path.join(src_dir, 'labels')
        if os.path.exists(label_dir):
            dst_label_dir = os.path.join(dst_dir, 'labels')
            os.makedirs(dst_label_dir, exist_ok=True)
            files = os.listdir(label_dir)
            if test_mode:
                files = files[:1]  # Take only the first file for testing
            for file in files:
                if file.endswith('.txt'):
                    shutil.copy2(
                        os.path.join(label_dir, file),
                        os.path.join(dst_label_dir, file)
                    )
                    if test_mode:
                        print(f"Copied test label to: {os.path.join(dst_label_dir, file)}")
    
    elif format == 'coco':
        # Copy COCO format annotations
        ann_dir = os.path.join(src_dir, 'annotations')
        if os.path.exists(ann_dir):
            dst_ann_dir = os.path.join(dst_dir, 'annotations')
            os.makedirs(dst_ann_dir, exist_ok=True)
            files = os.listdir(ann_dir)
            if test_mode:
                files = files[:1]  # Take only the first file for testing
            for file in files:
                if file.endswith('.json'):
                    shutil.copy2(
                        os.path.join(ann_dir, file),
                        os.path.join(dst_ann_dir, file)
                    )
                    if test_mode:
                        print(f"Copied test annotation to: {os.path.join(dst_ann_dir, file)}")

def test_conversion():
    """Test the conversion with a single image"""
    print("Running test conversion with a single image...")
    
    # Test YOLO format
    input_dir_yolo = "/Users/isha.yerramilli-rao/ir-person-detector/yolo_models/data/thermal/images/train_flir_images"
    yolo_output_dir = "/Users/isha.yerramilli-rao/ir-person-detector/yolo_models/data/thermal/grayscale/images/test"
    
    # Check input channels
    input_image_path = os.path.join(input_dir_yolo, os.listdir(input_dir_yolo)[0])
    input_image = Image.open(input_image_path)
    input_tensor = transforms.ToTensor()(input_image)
    print(f"Input image has {input_tensor.shape[0]} channels")
    
    dataloader_yolo = get_ir_dataloader(
        image_dir=input_dir_yolo,
        batch_size=1,
        num_workers=1,
        img_size=640,
        save_dir=yolo_output_dir,
        format='yolo',
        test_mode=True
    )
    
    print("\nTesting YOLO format conversion...")
    for i, batch in enumerate(dataloader_yolo):
        print(f"Processed test image, shape: {batch.shape}")
    
    copy_annotations(input_dir_yolo, yolo_output_dir, format='yolo', test_mode=True)
    print("YOLO test complete!")
    
    # Test COCO format
    input_dir_coco = "/Users/isha.yerramilli-rao/FLIR_ADAS_v2/images_thermal_train/data"
    coco_output_dir = "/Users/isha.yerramilli-rao/FLIR_ADAS_v2/grayscale/test"
    
    input_image_path = os.path.join(input_dir_coco, os.listdir(input_dir_coco)[0])
    input_image = Image.open(input_image_path)
    input_tensor = transforms.ToTensor()(input_image)
    print(f"Input image has shape {input_tensor.shape}")
    
    dataloader_coco = get_ir_dataloader(
        image_dir=input_dir_coco,
        batch_size=1,
        num_workers=1,
        img_size=640,
        save_dir=coco_output_dir,
        format='coco',
        test_mode=True
    )
    
    print("\nTesting COCO format conversion...")
    for i, batch in enumerate(dataloader_coco):
        print(f"Processed test image, shape: {batch.shape}")
    
    copy_annotations(input_dir_coco, coco_output_dir, format='coco', test_mode=True)
    print("COCO test complete!")

if __name__ == "__main__":
    # First run the test
    test_conversion()
    
    # Ask user if they want to proceed with full conversion
    response = input("\nTest complete! Would you like to proceed with full dataset conversion? (y/n): ")
    
    if response.lower() == 'y':
        input_dir_yolo = "/Users/isha.yerramilli-rao/ir-person-detector/yolo_models/data/thermal/images/train_flir_images"
        input_dir_coco = "/Users/isha.yerramilli-rao/FLIR_ADAS_v2/images_thermal_train/data"
        
        # Convert for YOLO format
        yolo_output_dir = "/Users/isha.yerramilli-rao/ir-person-detector/yolo_models/data/thermal/grayscale/images/train_flir_images"
        dataloader_yolo = get_ir_dataloader(
            image_dir=input_dir_yolo,
            batch_size=16, #processing 16 images at a time
            num_workers=4,
            img_size=640,
            save_dir=yolo_output_dir,
            format='yolo'
        )
        
        print("Converting images to yolo format")
        for i, batch in enumerate(dataloader_yolo):
            print(f"Processed YOLO batch {i+1}, shape: {batch.shape}")
        copy_annotations(input_dir_yolo, yolo_output_dir, format='yolo')
        print("YOLO complete!")
        
        # Convert for COCO format
        coco_output_dir = "/Users/isha.yerramilli-rao/FLIR_ADAS_v2/grayscale/images_thermal_train/data"
        dataloader_coco = get_ir_dataloader(
            image_dir=input_dir_coco,
            batch_size=16,
            num_workers=4,
            img_size=640,
            save_dir=coco_output_dir,
            format='coco'
        )
        
        print("\nConverting images to COCO format...")
        for i, batch in enumerate(dataloader_coco):
            print(f"Processed COCO batch {i+1}, shape: {batch.shape}")
        copy_annotations(input_dir_coco, coco_output_dir, format='coco')
        print("COCO complete!")