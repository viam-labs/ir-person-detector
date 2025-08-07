# ir-person-detector

This repository contains code for training and evaluating a person detection model using infrared video data from multiple streams. It uses Hydra to format the configs and easily navigate between different models. It is optimized for use on the Orin GPU. 
  
- Training pipeline for person detection using PyTorch and Ultralytics YOLOv8
- Utilities for:
  - Frame extraction from video
  - Visualization of bounding boxes
  - Dataset loading and augmentation
- Modular scripts for training, evaluation, and inference
## Models in this repo:

COCO models:
- Custom Detector with a CNN backbone
- [Efficient Net](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html)
- [FasterRCNN_MobileNet_V3_Large_FPN](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html)
- [SSDLite320_MobileNet_V3_Large](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html)

YOLO Model:
- [YOLov8n](https://docs.ultralytics.com/models/yolov8/)

To inspect model architectures, run <pre> python model_summary.py </pre>

## **File Organization:**

<pre>
## running COCO models 
coco_models/
├── configs
│   ├── config.yaml
│   ├── dataset
│   │   ├── flir.yaml
│   │   └── ir_data.yaml
│   ├── model
│   │   ├── custom_detector.yaml
│   │   ├── effnet.yaml
│   │   ├── faster_rcnn.yaml
│   │   └── ssdlite.yaml
│   └── optimization_results
│       ├── faster_rcnn.yaml
│       └── ssdlite.yaml  
└── src
    ├── datasets
    │   ├── flir_dataset.py
    │   ├── ir_dataset.py
    ├── eval.py
    ├── models
    │   ├── custom_detector.py
    │   ├── effnet_detector.py
    │   ├── faster_rcnn_detector.py
    │   └── ssdlite_detector.py
    ├── requirements.txt
    ├── train.py
    └── utils
        ├── clean_dataset.py
        └── transforms.py
## YOLO models
yolo_models/
├── configs
│   ├── config.yaml
│   ├── dataset
│   │   └── yolo.yaml
│   └── model
│       └── yolo.yaml
├── experiments
│   └── yolo_v8n_exp1_batchsize=16_in1_out5
└── src
    └── train_yolo.py
#datasets
├── FLIR_ADAS_v2 -> ../FLIR_ADAS_v2
├── ir_data -> ../ir_data
#preprocessing and misc scripts
├── filter.py
├── model_summary.py
├── requirements.txt
├── test_cuda.txt 
</pre>

## **Datasets:**

Two datasets were used throughout training, to compare results and optimally train. Both datasets include IR images of people, however the FLIR dataset included several other classes, which required filtering to isolate the 'person' class. The script filter_flir.py can be run to remove any other classes from the annotations files. The IR dataset included a higher number of clearer images of people, with many images of crowded scenes. 

1. [FLIR_ADAS dataset](https://adas-dataset-v2.flirconservator.com/#downloadguide)
2. [IR_data](https://www.kaggle.com/datasets/sikdermdsaiful/thermal-images-for-human-detection)

### **To get the dataset on your local device for training**

Download the dataset from kaggle, and run utils/clean_dataset.py with the path to the train, validation and test splits to produce a cleaned annotation files with the correct corresponding labels. 

Check the paths in datasets/ir_data.yaml to confirm that the paths to your image files and annotations are correct.

## **Training:**
Experimental outputs are saved in /multirun, organized according to date and time of the experiment. Config files in configs/optimization_results are saved from using Optuna hyperparameter tuning while training, and these tuned parameters can be used to override default settings for optimized train and val losses. 
The device is configurd to CUDA GPU in the setup configs for both COCO and YOLO, but can be changed to CPU if GPU is not available. 

### **To run the training pipeline on a COCO model:**

Run: 
<pre> bash python coco_models/src/train.py --multirun model=model_name optimization_results=model_name dataset=ir_data
</pre>
The model name and optimization_results options are custom_detector, effnet, ssdlite or faster_rcnn. The dataset options are ir_data or flir.

Any config parameters can be directly overriden from the terminal by adding the following at the end of the above command. <pre> ++param_name=override_value </pre>

At the end of training, output results will be saved in a folder corresponding to the date and time of the experiment in /multirun, with a best_model.pth file and tensorboard logging to monitor train and validation loss throughout training. Hydra overrides and the experiment config will be saved in multirun/hydra/. Training logs will be outputted in train.log in the same folder.

### **To run the training pipeline on a YOLO model:**

Required installation: 
<pre> pip install ultralytics </pre>

Run: <pre> bash python src/yolo_models/train_yolo.py --multirun model=model_name
</pre>
Experiment outputs will be saved in yolo_models/experiments/. 

## **Evaluating:**

### **For COCO models:**
Load the trained model (best_model.pth) path into src/coco_models/eval.py by editing the script.
<pre>
checkpoint_path = "path/to/best_model.pth"
</pre>

Run:
<pre> bash python src/coco_models/eval.py --multirun model=model_name
</pre>
The model name options are custom_detector, effnet, ssdlite or faster_rcnn. There is no need to override optimization results during evaluation as the config is not referenced during training. 

Outputs are saved in /outputs, with predictions.json and metrics.json which include the bounding boxes of predictions made by inference and the following metrics:
{
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl"
}

#### **Visualization:**

See outputs/visualizations folder. The first 5 processsed images will be saved, with ground truth boxes in green and prediction boxes in red, with annotated confidence scores. 

**Tensorboard:**
During training, loss values and other metrics are logged in tensorboard. To visualize them:

1. Install with <pre> pip install tensorboard </pre>

2. Launch a specific training run with <pre> tensorboard --logdir=path/to/tensorboard </pre>. Make sure to select "scalars" instead of "time series" data.

Example path/to/tensorboard: "outputs/2025-07-08/20-31-07/tensorboard" (do not enter the path to the actual file, but the tensorboard folder instead)

Tip: You can compare multiple runs by pointing --logdir to the parent directory containing several experiment folders.

### **For YOLO models:**
Evaluation metrics are computed by the ultralytics package, including a results.csv file, confusion matrices, mAP, precision and recall metrics and train/val batch loss diagrams. These are all saved in yolo_models/experiments when the train script is executed.

