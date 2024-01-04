# Resnet backbone: ResNet50, ResNet101, ResNet152
# Using Torchvision's pre-trained layers
#
# References
# ----------
# "Deep Residual Learning for Image Recognition" Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun


import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from enum import Enum
from math import ceil

from ..dataset import image
from .backbone import Backbone

class Architecture(Enum):
    """
    3 architecture based on Torchvision's pre-trained layers of Resnet
    """

    resnet50 = "resnet50"
    resnet101 = "resnet101"
    resnet152 = "resnet152"
    
def set_batchnorm_layers(module):
    if type(module) == nn.BatchNorm2d:
        module.eval()
    

class FeatureExtractor(nn.Module):
    """
    Base class for feature extractor, which extracts a feature map from the input image
    """

    def __init__(self, resnet):
        super().__init__()

        # Layers
        self._feature_extractor = nn.Sequential(
            resnet.conv1, # 0
            resnet.bn1, # 1
            resnet.relu, # 2
            resnet.maxpool, # 3
            resnet.layer1, # 4
            resnet.layer2, # 5
            resnet.layer3 # 6
        )

        # Freeze initial layers
        self._freeze(resnet.conv1)
        self._freeze(resnet.bn1)
        self._freeze(resnet.layer1)

        # Freeze all batchnorm layers
        self._freeze_batchnorm(self._feature_extractor)

    # Override nn.Module.train()
    def train(self, mode=True):
        super().train(mode)

        # During trainning, set all frozen blocks to evaluation mode
        if mode:
            self._feature_extractor.eval()
            self._feature_extractor[5].train()
            self._feature_extractor[6].train()

            self._feature_extractor.apply(set_batchnorm_layers)

    # Override nn.Module.forward
    def forward(self, x):
        """
        Args:
            x: iamge data, a tensor of [batch_size, channels, height, width]
        
        Returns:
            feature map, a tensor of [batch_size, 512, height//16, width//16]
        """

        x = self._feature_extractor(x)
        return x
    
    @staticmethod
    def _freeze(layer):
        for name, parameter in layer.named_parameters():
            parameter.requires_grad = False

    def _freeze_batchnorm(self, block):
        for module in block.modules():
            if type(module) == nn.BatchNorm2d:
                self._freeze(layer=module)

class FeatureVectorPooler(nn.Module):
    """
    Base class for feature vector pooler, which pools each ROI into a linear vector.
    """

    def __init__(self, resnet):
        super().__init__()

        # Layers
        self._layer4 = resnet.layer4

        # Freeze all batchnorm layers
        self._freeze_batchnorm(self._layer4)

    # Override nn.Module.train()
    def train(self, mode=True):
        super().train(mode)

        # During trainning, set all frozen blocks to evaluation mode
        if mode:
            self._layer4.apply(set_batchnorm_layers)
    
    # Override nn.Module.forward
    def forward(self, x):
        """
        Args:
            x: ROIs
        """

        x = self._layer4(x) # (N, 1024, 7, 7) -> (N, 2048, 4, 4)
        x = x.mean(-1).mean(-1) # -> (N, 2048)
        return x
    
    @staticmethod
    def _freeze(layer):
        for name, parameter in layer.named_parameters():
            parameter.requires_grad = False

    def _freeze_batchnorm(self, block):
        for module in block.modules():
            if type(module) == nn.BatchNorm2d:
                self._freeze(layer=module)
        
class ResnetBackbone(Backbone):
    """
    Class for resnet backbone models based on Torchvision
    """
    
    def __init__(self, architecture):
        super().__init__()
        assert architecture in [Architecture.resnet50, Architecture.resnet101, Architecture.resnet152]
        
        # Image preprocessing parameters
        self.feature_map_channels = 1024
        self.feature_pixels = 16
        self.feature_vector_size = 2048
        self.image_preprocess_params = image.PreprocessParams(channel_order = image.ChannelOrder.RGB, scaling = 1.0 / 255.0, means = [ 0.485, 0.456, 0.406 ], stds = [ 0.229, 0.224, 0.225 ])
        
        # Construct network model and pre-load ImageNet weights
        if architecture == Architecture.resnet50:
            resnet = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        elif architecture == Architecture.resnet101:
            resnet = torchvision.models.resnet101(weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
        elif architecture == Architecture.resnet152:
            resnet = torchvision.models.resnet152(weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
        print("Loaded IMAGENET1K_V1 pre-trained weights for Torchvision %s backbone" % architecture.value)
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(resnet=resnet)
        # Feature vector pooler
        self.feature_vector_pooler = FeatureVectorPooler(resnet=resnet)
        
    def get_backbone_shape(self, image_shape):
        """
        Given the shape of input image, compute the shape of each stage pf the backbone network.

        Args:
            image_shape: Tuple[channels(int), height(int), width(int)]

        Returns:
            feature_map_shape: uple[channels(int), height(int), width(int)]
        """
        
        return (self.feature_map_channels, ceil(image_shape[-2]/self.feature_pixels), ceil(image_shape[-1]/self.feature_pixels))