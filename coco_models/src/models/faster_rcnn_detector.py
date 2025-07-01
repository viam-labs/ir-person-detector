import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FasterRCNN
from omegaconf import DictConfig
import logging
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.mobilenetv3 import mobilenet_v3_large

log = logging.getLogger(__name__)
#resizes to 800x1333 (will resize any image to this size)
class FasterRCNNDetector(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(FasterRCNNDetector, self).__init__()
        
        backbone = mobilenet_v3_large(weights=None)
        backbone.features[0][0] = replace_first_conv_to_1channel(backbone.features[0][0])

        backbone_fpn = BackboneWithFPN(backbone.features,
                                        return_layers={'4': '0', '6': '1', '12': '2', '16': '3'},  # typical MobileNetV3 FPN layers
                                        in_channels_list=[40, 40, 112, 960],
                                        out_channels=256)
                                        
        
        # Load pretrained model withOUT default weights
        self.model = FasterRCNN(
            backbone= backbone_fpn,
            num_classes=cfg.model.num_classes +1, #automatically adds "background" class through RPN 
            box_nms_thresh=0.5,     # NMS IoU threshold
            box_detections_per_img=10  # Max detections per image
        )
        
        input_h, input_w = cfg.model.transform.input_size
        self.model.transform = SingleChannelRCNNTransform(
            min_size= input_h, #can adjust accordingly 
            max_size=input_w,
            image_mean=[0.485],
            image_std=[0.229]
        )
    
    def forward(self, data, targets=None):
        outputs = self.model(data, targets)
        return outputs

class SingleChannelRCNNTransform(GeneralizedRCNNTransform):
    def __init__(self, min_size, max_size, image_mean, image_std):
        # Override mean and std for single channel
        image_mean = [image_mean[0]]  
        image_std = [image_std[0]]   
        super().__init__(min_size, max_size, image_mean, image_std)

def replace_first_conv_to_1channel(conv3: nn.Conv2d) -> nn.Conv2d:
    new_conv = nn.Conv2d( #rebuilding first conv layer to accept 1 channel input 
        in_channels=1,
        out_channels=conv3.out_channels,
        kernel_size=conv3.kernel_size,
        stride=conv3.stride,
        padding=conv3.padding,
        bias=(conv3.bias is not None)
    )
    # Custom init
    nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
    if new_conv.bias is not None:
        nn.init.zeros_(new_conv.bias)
    return new_conv