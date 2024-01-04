# Backone base class for wrapping backbone models that provide:
# 1. Feature Extractor:
# Produce a feature map from a given image, which is used in the RPN and detecion.
# 2. Feature vector pooler:
# Convert each ROIs into a linear feature vector.

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from ..dataset import image

class Backbone:
    """
    Backbone base class, overrided for backbone models.
    """

    def __init__(self):
        self.feature_map_channels = 0
        self.feature_pixels = 0
        self.feature_vector_size = 0
        self.iamge_preprocess_params = image.PreprocessParams(channel_order=image.ChannelOrder.BGR, scaling=1.0, means=[103.939, 116.779, 123.680], stds=[1, 1, 1])

        self.feature_extractor = None # nn.Module a convert input image to a feature map
        self.feature_vector_pooler = None # nn.Module convert a feature map to a linear feature vector

    def get_backbone_shape(self, image_shape):
        """
        Given the shape of input image, compute the shape of each stage pf the backbone network.

        Args:
        image_shape: Tuple[channels(int), height(int), width(int)]

        Returns:
        feature_map_shape: Tuple[channels(int), height(int), width(int)]
        """

        return image_shape[-3: ]

        