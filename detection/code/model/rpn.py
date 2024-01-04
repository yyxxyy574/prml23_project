# Region Proposal Network
# Given a feature map, generate objectness scores for each anchor box, and boxes in the form of modifications to anchor points and dimensions.
# Use only a single output per anchor and sigmoid activation.

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms

from . import math_utils

def extract_top_n(list, number):
    return list[0: number]

def clip_bbox(bboxes, image_shape):
    bboxes[:, 0: 2] = torch.clamp(bboxes[:, 0: 2], min=0)
    bboxes[:, 2] = torch.clamp(bboxes[:, 2], max=image_shape[1])
    bboxes[:, 3] = torch.clamp(bboxes[:, 3], max=image_shape[2])
    return bboxes

class RPN(nn.Module):
    """
    Build the model of Region Proposal Network, which predicts objectness scores and regress region-of-interest box proposals given an input feature map.
    
    Args:
        feature_map_channels: int
        exclude_edge_proposals: bool
    """
    
    def __init__(self, feature_map_channels, exclude_edge_proposals):
        super().__init__()

        # Layers
        # Initialize weights
        anchor_num = 9
        self._rpn_conv1 = nn.Conv2d(in_channels=feature_map_channels, out_channels=feature_map_channels, kernel_size=(3, 3), stride=1, padding="same")
        self._rpn_conv1.weight.data.normal_(mean=0.0, std=0.01)
        self._rpn_conv1.bias.data.zero_()
        self._rpn_classifier = nn.Conv2d(in_channels=feature_map_channels, out_channels=anchor_num, kernel_size=(1, 1), stride=1, padding="same")
        self._rpn_classifier.weight.data.normal_(mean=0.0, std=0.01)
        self._rpn_classifier.bias.data.zero_()
        self._rpn_bbox = nn.Conv2d(in_channels=feature_map_channels, out_channels=anchor_num * 4, kernel_size=(1, 1), stride=1, padding="same")
        self._rpn_bbox.weight.data.normal_(mean=0.0, std=0.01)
        self._rpn_bbox.bias.data.zero_()

        # Whether to exclude edge proposals or not
        self._exclude_edge_proposals = exclude_edge_proposals
        

    def forward(self, x, image_shape, anchors, anchors_vaild_map, before_proposal_num, after_proposal_num):
        """
        Args:
            x: feature_map, torch.Tensor(batch_size, channels, height, width)
            image_shape: Tuple[channels(int), height(int), width(int)]
            anchors: np.array(height, width, [center_y, center_x, height, width] * anchor_num)
            anchors_vaild_map: np.array(height, width, anchor_num)
            before_proposal_num: int, sorted by objectness score, hwo many of the best proposals to extract before non-maximum suppression
            after_proposal_num: int, sorted by objectness score, how many of the best proposals to choose after non-maximum suppression

        Returns:
            rpn_objectness_scores: torch.Tensor(batch_size, height, width, anchor_num)
            rpn_bbox_regressions: torch.Tensor(batch_size, height, width, deltas to be applied to anchors[dy, dx, dh, dw] * anchor_num)
            rpn_bbox_proposals: torch.Tensor(corresponding proposed bounding box corners[y1, x1, y2, x2] * after_proposal_num)
        """

        # Pass through the network
        x = F.relu(self._rpn_conv1(x))
        rpn_objectness_scores = torch.sigmoid(self._rpn_classifier(x))
        rpn_bbox_regressions = self._rpn_bbox(x)
        # Change the shape
        rpn_objectness_scores = rpn_objectness_scores.permute(0, 2, 3, 1).contiguous()
        rpn_bbox_regressions = rpn_bbox_regressions.permute(0, 2, 3, 1).contiguous()
        
        # Extract anchors and bbox regressions as (N, 4) torch.Tensor and objectness scores as (N, ) list
        anchors, objectness_socres, bbox_regressions = self._extract_valid_anchors(anchors=anchors,
                                                                                   anchors_vaild_map=anchors_vaild_map,
                                                                                   objectness_socres=rpn_objectness_scores,
                                                                                   bbox_regressions=rpn_bbox_regressions)

        # Detach from graph to avoid backprop
        bbox_regressions = bbox_regressions.detach()

        # Computer bbox corners from anchors and corresponding regressions
        rpn_bbox_proposals = math_utils.compute_bbox_torch(
            anchors=torch.from_numpy(anchors).cuda(),
            box_deltas=bbox_regressions,
            box_delta_means=torch.tensor([0, 0, 0, 0], dtype=torch.float32, device="cuda"),
            box_delta_stds=torch.tensor([1, 1, 1, 1], dtype=torch.float32, device="cuda") 
        )

        # Keep the top-before_proposal_num-socre proposals
        sorted_indices = torch.argsort(objectness_socres)
        sorted_indices = sorted_indices.flip(dims=(0, ))
        rpn_bbox_proposals = extract_top_n(rpn_bbox_proposals[sorted_indices], before_proposal_num)
        objectness_socres = extract_top_n(objectness_socres[sorted_indices], before_proposal_num)

        # Clip to image boundaries
        rpn_bbox_proposals = clip_bbox(rpn_bbox_proposals, image_shape)

        # Exclude proposals that are less than 16 pixels on a side
        height = rpn_bbox_proposals[:, 2] - rpn_bbox_proposals[:, 0]
        width = rpn_bbox_proposals[:, 3] - rpn_bbox_proposals[:, 1]
        idxs = torch.where((height >= 16) & (width >= 16))[0]
        rpn_bbox_proposals = rpn_bbox_proposals[idxs]
        objectness_socres = objectness_socres[idxs]

        # Perform non-maximum suppression
        idxs = nms(boxes=rpn_bbox_proposals, scores=objectness_socres, iou_threshold=0.7)
        idxs = extract_top_n(idxs, after_proposal_num)
        rpn_bbox_proposals = rpn_bbox_proposals[idxs]
        
        return rpn_objectness_scores, rpn_bbox_regressions, rpn_bbox_proposals

    def _extract_valid_anchors(self, anchors, anchors_vaild_map, objectness_socres, bbox_regressions):
        # Only one batch
        assert objectness_socres.shape[0] == 1

        height, width, anchor_num = anchors_vaild_map.shape
        anchors = anchors.reshape((height * width * anchor_num, 4))
        anchors_vaild_map = anchors_vaild_map.reshape((height * width * anchor_num))
        objectness_socres = objectness_socres.reshape((height * width * anchor_num))
        bbox_regressions = bbox_regressions.reshape((height * width * anchor_num, 4))

        if self._exclude_edge_proposals:
            # Filter out proposals that are generated at invalid anchors
            idxs = anchors_vaild_map > 0
            return anchors[idxs], objectness_socres[idxs], bbox_regressions[idxs]
        else:
            # Keep all proposals
            return anchors, objectness_socres, bbox_regressions