#ended up not using this yet in order to focus on flir --> will use if need to finetune using roboflow

from PIL import Image
import os
import shutil

def pad_and_resize_image(image, target_size=(640, 640), fill_color=0):
    """Resize and pad image to target size while maintaining aspect ratio."""
    original_size = image.size  # (width, height)
    ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    resized = image.resize(new_size, Image.ANTIALIAS)

    # Create new image and paste resized on center --> computationally expensive (not a good idea)
    new_im = Image.new("L", target_size, color=fill_color)
    top_left = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
    new_im.paste(resized, top_left)
    return new_im, ratio, top_left

def adjust_yolo_labels(label_path, output_path, ratio, pad, orig_size):
    """Adjust YOLO labels according to new size and padding."""
    if not os.path.exists(label_path):
        return

    with open(label_path, "r") as f:
        lines = f.readlines()

    adjusted = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = map(float, parts)

        # Convert back to absolute using original size
        abs_x = x * orig_size[0]
        abs_y = y * orig_size[1]
        abs_w = w * orig_size[0]
        abs_h = h * orig_size[1]

        # Resize + pad coordinates
        abs_x = abs_x * ratio + pad[0]
        abs_y = abs_y * ratio + pad[1]
        abs_w = abs_w * ratio
        abs_h = abs_h * ratio

        # Normalize to new 640x640 size
        new_x = abs_x / 640
        new_y = abs_y / 640
        new_w = abs_w / 640
        new_h = abs_h / 640

        adjusted.append(f"{int(cls)} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}")

    with open(output_path, "w") as f:
        for line in adjusted:
            f.write(line + "\n")

def process_dataset(image_dir, label_dir, out_image_dir, out_label_dir):
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    for file in os.listdir(image_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_dir, file)
            label_path = os.path.join(label_dir, file.rsplit(".", 1)[0] + ".txt")

            img = Image.open(image_path).convert("L")
            orig_size = img.size
            resized_img, ratio, pad = pad_and_resize_image(img)

            resized_img.save(os.path.join(out_image_dir, file))

            if os.path.exists(label_path):
                out_label_path = os.path.join(out_label_dir, os.path.basename(label_path))
                adjust_yolo_labels(label_path, out_label_path, ratio, pad, orig_size)
            else:
                # If no label file exists, create empty one
                open(os.path.join(out_label_dir, os.path.basename(label_path)), "w").close()

# === CONFIG ===
base_path = "/path/to/your/dataset"  # e.g., YOLO/data/thermal
splits = ["train", "val", "test"]

for split in splits:
    process_dataset(
        image_dir=os.path.join(base_path, "images", f"{split}_flir_images"),
        label_dir=os.path.join(base_path, "labels", f"{split}_flir_images"),
        out_image_dir=os.path.join(base_path, "images", f"{split}_flir_640"),
        out_label_dir=os.path.join(base_path, "labels", f"{split}_flir_640")
    )

pad_and_resize_image(
    input_dir="/path/to/FLIR/images/train",    
    output_dir="/path/to/resized/images/train",  
    size=(640, 640)
)
