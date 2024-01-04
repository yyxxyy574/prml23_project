# Detect bboxes and classifications from a series of proposals

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import RoIPool

class DetectionNetwork(nn.Module):
    """
    Given a series proposals, ie, ROIs, pool into linear vectors, to produce classifications and bboxes.

    Args:
        class_num: int, the number of dataset classes
        backbone: the backbone model for pooling feature vector
    """

    def __init__(self, class_num, backbone):
        super().__init__()

        # Layers
        # Initialize weights
        self._roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1.0 / backbone.feature_pixels)
        self._feature_vector_pooler = backbone.feature_vector_pooler
        self._classifier = nn.Linear(in_features=backbone.feature_vector_size, out_features=class_num)
        self._classifier.weight.data.normal_(mean=0.0, std=0.01)
        self._classifier.bias.data.zero_()
        self._regressor = nn.Linear(in_features=backbone.feature_vector_size, out_features=(class_num - 1) * 4) # Exclude the background
        self._regressor.weight.data.normal_(mean=0.0, std=0.01)
        self._regressor.bias.data.zero_()

    def forward(self, x, y):
        """
        Args:
            x: feature_map, torch.Tensor(batch_size, channels, height, width)
            y: rois, torch.Tensor(roi_num, [y1, x1, y2, x2] * roi_num)

        Returns:
            detected_classes: torch.Tensor(roi_num, class_num)
            detected_bbox_regressions: torch.Tensor(roi_num, [dy, dx, dh, dw] * (class_num - 1))
        """

        # Only one batch
        batch_idxs = torch.zeros((y.shape[0], 1)).cuda()

        # Change proposal forms
        # each row -> (batch_idx, x1, y1, x2, y2)
        y = torch.cat([batch_idxs, y], dim=1)
        y = y[:, [0, 2, 1, 4, 3]]

        # Perform RoI pooling
        y = self._roi_pool(x, y)

        # Forward propagate
        y = self._feature_vector_pooler(y)
        detected_classes = self._classifier(y)
        detected_classes = F.softmax(detected_classes, dim=1)
        detected_bbox_regressions = self._regressor(y)

        return detected_classes, detected_bbox_regressions