# VGG16 backbone
# Using Torchvision's pre-trained layers

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from math import ceil

from ..dataset import image
from .backbone import Backbone

class FeatureExtractor(nn.Module):
    """
    Base class for feature extractor, which extracts a feature map from the input image
    """
    
    def __init__(self, vgg16):
        super().__init__()
        assert len(vgg16.features)==31 and type(vgg16.features[-1])==nn.modules.pooling.MaxPool2d
        
        # Layers
        self._layers = vgg16.features[0: -1] # All feature extractor layers except the MaxPool2d
        
        # Freeze the first 4 Conv2d layers
        i = 0
        for layer in self._layers:
            if type(layer) == nn.Conv2d and i < 4:
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False
                i += 1
                
    def forward(self, x):
        """
        Args:
            x: iamge data, a tensor of [batch_size, channels, height, width]
        
        Returns:
            feature map, a tensor of [batch_size, 512, height//16, width//16]
        """

        return self._layers(x)
    
class FeatureVectorPooler(nn.Module):
    """
    Base class for feature vector pooler, which pools each ROI into a linear vector.
    """
    
    def __init__(self, vgg16):
        super().__init__()
        assert len(vgg16.classifier)==7 and type(vgg16.classifier[-1])==nn.modules.linear.Linear
        
        # Classifier layers
        self._layers = vgg16.classifier[0: -1] # All classifier layers except the last one
        
    def forward(self, x):
        """
        Args:
            x: ROIs
        """
        
        x = x.reshape((x.shape[0], 512 * 7 * 7))
        x = self._layers(x)
        return x

class VGG16Backbone(Backbone):
    """
    Class for vgg16 backbone model based on Torchvision
    """
    
    def __init__(self, dropout):
        super().__init__()
        
        # Image preprocessing parameters
        self.feature_map_channels = 512
        self.feature_pixels = 16
        self.feature_vector_size = 4096
        self.image_preprocess_params = image.PreprocessParams(channel_order = image.ChannelOrder.RGB, scaling = 1.0 / 255.0, means = [ 0.485, 0.456, 0.406 ], stds = [ 0.229, 0.224, 0.225 ])
        
        # Construct network model and pre-load ImageNet weights
        vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1, dropout=dropout)
        print("Loaded IMAGENET1K_V1 pre-trained weights for Torchvision VGG-16 backbone")
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(vgg16=vgg16)
        # Feature vector pooler
        self.feature_vector_pooler = FeatureVectorPooler(vgg16=vgg16)
        
    def get_backbone_shape(self, image_shape):
        """
        Given the shape of input image, compute the shape of each stage pf the backbone network.

        Args:
            image_shape: Tuple[channels(int), height(int), width(int)]

        Returns:
            feature_map_shape: uple[channels(int), height(int), width(int)]
        """
    
        return (self.feature_map_channels, ceil(image_shape[-2] // self.feature_pixels), ceil(image_shape[-1] // self.feature_pixels))