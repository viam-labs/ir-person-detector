from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import json
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from hydra.utils import get_original_cwd
import torch
import numpy as np
from models.custom_detector import ThermalDetector
from models.faster_rcnn_detector import FasterRCNNDetector
from models.effnet_detector import EfficientNetDetector
from models.ssdlite_detector import SSDLiteDetector
from datasets.ir_dataset import IRDataset
from utils.transforms import build_transforms, GPUCollate
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

log = logging.getLogger(__name__)

def visualize_predictions(image,predictions,targets, title="", output_dir=None):
    img_np = image.cpu().numpy()[0]  # 1 channel for grayscale
    fig, ax = plt.subplots(1)
    ax.imshow(img_np, cmap='gray')
    
    # Plot predicted boxes in red
    if predictions is not None and len(predictions['boxes']) > 0:
        for box, score in zip(predictions['boxes'], predictions['scores']):
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                  edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{score:.2f}', color='red')
    
    # Plot ground truth boxes in green
    if targets is not None and targets['boxes'].numel() > 0:
        boxes = targets['boxes'].view(-1, 4)
        for box in boxes:
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                  edgecolor='g', facecolor='none')
            ax.add_patch(rect)
    
    plt.title(title)
    plt.axis('off')
    
    # Save figure 
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{title.replace(' ', '_')}.png"
        plt.savefig(save_path)
    plt.close()  

def evaluate_model(model, data_loader, cfg: DictConfig):
    """
    Evaluate model on test set and compute COCO metrics
    """
    model.eval()
    results = []
    total_predictions = 0
    total_boxes = 0
    
    # save in visualizations directory
    vis_dir = Path(cfg.logging.save_dir) / "visualizations"
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(data_loader)):
            predictions = model(data)
            
            # Visualize first n images
            if batch_idx == 0:
                for i in range(min(5, len(data))):
                    visualize_predictions(
                        data[i], 
                        predictions[i],
                        targets[i],
                        title=f"Image {targets[i]['image_id']}",
                        output_dir=vis_dir
                    )
                        
            for pred, target in zip(predictions, targets):
                image_id = target['image_id'].item()
                boxes = pred['boxes']
                scores = pred['scores']
                total_predictions += 1
                total_boxes += len(boxes)
    
                if len(boxes) > 0:
                    # convert from [x1,y1,x2,y2] to COCO format [x,y,w,h]
                    boxes_coco = torch.zeros_like(boxes)
                    boxes_coco[:, 0] = boxes[:, 0]  # x
                    boxes_coco[:, 1] = boxes[:, 1]  # y
                    boxes_coco[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
                    boxes_coco[:, 3] = boxes[:, 3] - boxes[:, 1]  # h
                    
                    # Add all detections for this image
                    results.extend([
                        {
                            'image_id': image_id,
                            'category_id': 1,  # person
                            'bbox': box.tolist(),
                            'score': score.item()
                        }
                        for box, score in zip(boxes_coco, scores)
                    ])
    log.info(f"Evaluation complete. Total predictions: {total_predictions}")
    log.info(f"Total boxes detected: {total_boxes}")
    return results

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    base_dir = Path(get_original_cwd())
    mp.set_start_method('spawn', force=True)
    device = cfg.model.device
    
    #Create model
    if cfg.model.name == "custom_detector":
        model = ThermalDetector(cfg).to(device)
    elif cfg.model.name == "faster_rcnn":
        model = FasterRCNNDetector(cfg).to(device)
    elif cfg.model.name == "effnet":
        model = EfficientNetDetector(cfg).to(device)
    elif cfg.model.name == "ssdlite":
        model = SSDLiteDetector(cfg).to(device)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.name}")
 
    checkpoint_path = "/root/ir-person-detector/multirun/2025-07-01/22-21-57/11/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_transform = build_transforms(cfg, is_train=False, test=True)

    # Create test dataset and dataloader
    test_dataset = IRDataset(
        json_file= Path(cfg.dataset.data.test_annotations),
        thermal_dir= Path(cfg.dataset.data.test_images),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers= cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        collate_fn=GPUCollate(device, test_transform)
    )

    results = evaluate_model(model, test_loader, cfg)

    # saving predictions
    output_dir = Path(cfg.logging.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = output_dir / f"{cfg.model.name}_predictions.json"
    with open(predictions_file, "w") as f:
        json.dump(results, f)

    # COCO metrics
    gt_path = Path(cfg.dataset.data.test_annotations)
    with open(predictions_file, 'r') as f:
        pred_data = json.load(f)
    log.info(f"no of predictions: {len(pred_data)}")
    
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    log.info(f"ground truth images: {len(gt_data['images'])}")
    log.info(f"ground truth annotations: {len(gt_data['annotations'])}")

    # Check if there are any matching image IDs
    pred_img_ids = set(p['image_id'] for p in pred_data)
    gt_img_ids = set(ann['image_id'] for ann in gt_data['annotations'])
    matching_ids = pred_img_ids.intersection(gt_img_ids)

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(str(predictions_file))
    
    coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt)
    coco_eval.params.catIds = [1]
    coco_eval.params.iouType = 'bbox'
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Save metrics
    metrics = { #AP = average precision
        'AP': coco_eval.stats[0],  # AP at IoU=0.50:0.95
        'AP50': coco_eval.stats[1],  # AP at IoU=0.50
        'AP75': coco_eval.stats[2],  # AP at IoU=0.75
        'APs': coco_eval.stats[3],   # AP for small objects
        'APm': coco_eval.stats[4],   # AP for medium objects
        'APl': coco_eval.stats[5],   # AP for large objects
    }
    
    metrics_file = output_dir / f"{cfg.model.name}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    
    log.info(f"AP50: {metrics['AP50']:.3f}")

if __name__ == "__main__":
    main()