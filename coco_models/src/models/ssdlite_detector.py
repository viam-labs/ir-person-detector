import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection import _utils as det_utils
import hydra
from omegaconf import DictConfig
import logging
log = logging.getLogger(__name__)
#requires image of size 320x320

class SSDLiteDetector(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(SSDLiteDetector, self).__init__()
        
        # Load pretrained model withOUT default weights
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=None)

        # Get number of anchors
        num_anchors = self.model.anchor_generator.num_anchors_per_location()
        size = (320, 320)
        in_channels = det_utils.retrieve_out_channels(self.model.backbone, size) #retrievng out channels from backbone --> last layer of backbone is first layer of classification head (320 is the expected size for ssdlite)
        # Create new classification head
        self.model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=cfg.model.num_classes + 1
        )
        
        self.transforms = weights.transforms()

    def forward(self, data, targets=None):
        return self.model(data, targets)

    