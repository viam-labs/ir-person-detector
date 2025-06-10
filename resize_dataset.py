from PIL import Image
import os

input_dir = "/Users/isha.yerramilli-rao/FLIR_ADAS_v2/images_thermal_train"
output_dir = "/Users/isha.yerramilli-rao/FLIR_ADAS_v2/images_thermal_train"
os.makedirs(output_dir, exist_ok=True)

target_size = (640, 640)

for fname in os.listdir(input_dir):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = Image.open(os.path.join(input_dir, fname))
        img_resized = img.resize(target_size, Image.BILINEAR)
        img_resized.save(os.path.join(output_dir, fname))