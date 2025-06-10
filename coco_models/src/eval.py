#CHEKC THIS FILE FOR EVALUATION (UPDATE MIGHT BE REQUIRED)
#FOR YOLO FORMAT 

# import os
# import torch
# import torchvision
# from torchvision.transforms import functional as F
# from PIL import Image
# import cv2
# import numpy as np
# from sklearn.metrics import precision_score, recall_score, f1_score
# from tqdm import tqdm

# # Load pretrained Faster R-CNN model
# model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
# model.eval()

# # Use CPU or GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# IMAGE_DIR = "data/thermal/images/val_flir_images"
# LABEL_DIR = "data/thermal/labels/val_flir_images"
# IMG_SIZE = (640, 512)  # original FLIR resolution

# #load YOLO labels
# def load_yolo_labels(label_path, img_w, img_h):
#     boxes = []
#     labels = []
#     if not os.path.exists(label_path):
#         return [], []

#     with open(label_path, "r") as f:
#         for line in f.readlines():
#             cls, x, y, w, h = map(float, line.strip().split())
#             # YOLO coords: center x, y, width, height → absolute corners
#             cx, cy, bw, bh = x * img_w, y * img_h, w * img_w, h * img_h
#             x1 = cx - bw / 2
#             y1 = cy - bh / 2
#             x2 = cx + bw / 2
#             y2 = cy + bh / 2
#             boxes.append([x1, y1, x2, y2])
#             labels.append(int(cls))
#     return boxes, labels

# def iou(boxA, boxB):
#     # Intersection over Union
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#     inter = max(0, xB - xA) * max(0, yB - yA)

#     boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
#     union = boxAArea + boxBArea - inter
#     return inter / union if union > 0 else 0

# def evaluate_detection(results, iou_thresh=0.5, conf_thresh=0.5):
#     y_true = []
#     y_pred = []

#     for res in results:
#         gt_boxes = res["gt_boxes"]
#         pred_boxes = res["pred_boxes"]
#         pred_scores = res["pred_scores"]

#         pred_boxes_filtered = [box for box, score in zip(pred_boxes, pred_scores) if score >= conf_thresh]

#         matched = [False] * len(gt_boxes)

#         for pb in pred_boxes_filtered:
#             found_match = False
#             for i, gb in enumerate(gt_boxes):
#                 if not matched[i] and iou(pb, gb) >= iou_thresh:
#                     matched[i] = True
#                     found_match = True
#                     break
#             y_pred.append(1 if found_match else 0)

#         y_true += [1] * len(gt_boxes)
#         y_pred += [1] * sum(matched) + [0] * (len(pred_boxes_filtered) - sum(matched))

#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#     accuracy = sum([yt == yp for yt, yp in zip(y_true, y_pred)]) / len(y_true)

#     return precision, recall, f1, accuracy


# # evaluation Loop
# results = []
# with torch.no_grad():
#     for img_file in tqdm(os.listdir(IMAGE_DIR)):
#         if not img_file.lower().endswith((".jpg", ".png")):
#             continue

#         # Load image and convert to RGB
#         path = os.path.join(IMAGE_DIR, img_file)
#         img = Image.open(path).convert("L")  # grayscale
#         img_rgb = img.convert("RGB")
#         img_tensor = F.to_tensor(img_rgb).to(device)

#         # Load GT labels
#         label_file = os.path.join(LABEL_DIR, os.path.splitext(img_file)[0] + ".txt")
#         gt_boxes, gt_labels = load_yolo_labels(label_file, *img.size)

#         # Inference
#         outputs = model([img_tensor])[0]
#         pred_boxes = outputs["boxes"].cpu().numpy()
#         pred_scores = outputs["scores"].cpu().numpy()
#         pred_labels = outputs["labels"].cpu().numpy()

#         # Store per-image result for later eval (IoU, precision, etc.)
#         results.append({
#             "image_id": img_file,
#             "gt_boxes": gt_boxes,
#             "pred_boxes": pred_boxes,
#             "pred_scores": pred_scores,
#             "pred_labels": pred_labels,
#         })

# # print one example result to check 
# sample = results[0]

# print(f"Image: {sample['image_id']}")
# print(f"GT Boxes: {sample['gt_boxes']}")
# print(f"Predicted Boxes: {sample['pred_boxes'][:3]}")
# print(f"Scores: {sample['pred_scores'][:3]}")
# #print evaluation metrics 
# p, r, f1, acc = evaluate_detection(results)
# print(f"Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}, Accuracy: {acc:.3f}")

#FOR COCO FORMAT

import os
import torch
import torchvision
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import numpy as np
import json
from PIL import Image
import pickle

with open("flir_coco_index.pkl", "wb") as f:
    pickle.dump(COCO, f)

IMG_DIR = "/Users/isha.yerramilli-rao/FLIR_ADAS_v2/images_thermal_train"
ANN_FILE = "/Users/isha.yerramilli-rao/FLIR_ADAS_v2/images_thermal_train/coco.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load model
model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
model.to(DEVICE)
model.eval()

# load dataset (flir, COCO)
class CocoThermal(CocoDetection):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # triplicateto fill RGB (requires 3 channels)
        img = img.convert("L")  # make sure it's grayscale
        img_tensor = F.to_tensor(img).repeat(3, 1, 1).to(DEVICE)  # [1, H, W] → [3, H, W]
        image_id = self.ids[index]
        target_dict = {"image_id": image_id}
        return img_tensor, target_dict

dataset = CocoThermal(IMG_DIR, ANN_FILE)

def load_coco_index(json_path, pickle_path="flir_coco_index.pkl"):

    if os.path.exists(pickle_path):
        print(f"Loading cached COCO index from {pickle_path}...")
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    else:
        print("Building COCO index from JSON...")
        coco = COCO(json_path)
        with open(pickle_path, "wb") as f:
            pickle.dump(coco, f)
        return coco


# Inference loop
coco = load_coco_index(ANN_FILE)
results = []

with torch.no_grad():
    for idx in tqdm(range(len(dataset))):
        img_tensor, meta = dataset[idx]
        image_id = meta["image_id"]

        outputs = model([img_tensor])[0]
        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score < 0.3:
                continue
            x1, y1, x2, y2 = box
            results.append({
                "image_id": int(image_id),
                "category_id": int(label),  # COCO: person=1 (only running on person detection) --> want to filter to just use 1
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score)
            })

# saving pred to coco json
with open("fasterrcnn_predictions.json", "w") as f:
    json.dump(results, f)

# Evaluate using COCOeval
coco_pred = coco.loadRes("fasterrcnn_predictions.json")
coco_eval = COCOeval(coco, coco_pred, iouType="bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
