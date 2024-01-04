# Faster RCNN class
# Combine each component into a whole aritecture
# Build corresponding training and inference models

import numpy as np
import torch
from torch import nn
from torchvision.ops import nms
import random
from dataclasses import dataclass

from detection.code import utils
from . import anchor
from . import detector
from . import loss
from . import resnet
from . import vgg16
from . import rpn
from . import math_utils

class FasterRCNN(nn.Module):
    
    @dataclass
    class Loss:
        rpn_class: float
        rpn_bbox: float
        detector_class: float
        detector_bbox: float
        total: float
    
    def __init__(self, class_num, backbone, rpn_minibatch_size=256, proposal_batch_size=128, exclude_edge_proposals=False):
        """
        Args:
            class_num: int, number of output classes
            backbone: backbone network for extracting feature and pooling feature vector
            rpn_minibatch_size: int, number of ground truth anchors sampled for training at each step
            proposal_batch_size: int, number of region proposals to sample at each training step
            exclude_edge_proposals: bool, whether to exclude proposals generated at invalid anchors (those that straddle image edges)
        """
        super().__init__()
        
        self._class_num = class_num
        self._rpn_minibatch_size = rpn_minibatch_size
        self._proposal_batch_size = proposal_batch_size
        self._detector_bbox_delta_means = [0, 0, 0, 0]
        self._detector_bbox_delta_stds = [0.1, 0.1, 0.2, 0.2]
        
        self._backbone = backbone
        self._feature_extractor = backbone.feature_extractor
        self._rpn = rpn.RPN(feature_map_channels=backbone.feature_map_channels, exclude_edge_proposals=exclude_edge_proposals)
        self._detector = detector.DetectionNetwork(class_num=class_num, backbone=backbone)
    
    def forward(self, image_data, anchors=None, anchors_vaild_map=None):
        """
        Forward inference, uesed for test and evaluation

        Args:
            image_data: torch.Tensor(batch_size, channels, height, width)
            anchors: torch.Tensor(height, width, [center_y, center_x, height, width] * anchor_num)
            anchors_vaild_map: torch.Tensor(height, width, anchor_num)
            
        Returns:
            proposals: np.array(proposal_num, [y1, x1, y2, x2] * proposal_num) from RPN
            classes: torch.Tensor(roi_num, class_num) from detection network
            regressions: torch.Tensor(roi_num, [dy, dx, dh, dw] * (class_num - 1)) from detection network
        """
        
        assert image_data.shape[0] == 1
        
        image_shape = image_data.shape[1: ]
        
        # If missing, compute anchors and anchor_vaild_map
        if anchors is None or anchors_vaild_map is None:
            feature_map_shape = self._backbone.get_backbone_shape(image_shape=image_shape)
            anchors, anchors_vaild_map = anchor.generate_anchors(image_shape=image_shape, feature_map_shape=feature_map_shape, feature_pixels=self._backbone.feature_pixels)
            
        # Forward inferring successively
        feature_map = self._feature_extractor(image_data)
        objectness_scores, bbox_regressions, proposals = self._rpn(
            feature_map,
            image_shape=image_shape,
            anchors=anchors,
            anchors_vaild_map=anchors_vaild_map,
            before_proposal_num=6000,
            after_proposal_num=300
        )
        classes, regressions = self._detector(feature_map, proposals)
        
        return proposals, classes, regressions
    
    @ utils.no_grad
    def predict(self, image_data, score_threshold, anchors=None, anchors_vaild_map=None):
        """
        Predict bboxes and classes for an image

        Args:
            image_data: torch.Tensor(batch_size, channels, height, width)
            socre_threshold: float, minimum required score threshold for a detection to be considered
            anchors: torch.Tensor(height, width, [center_y, center_x, height, width] * anchor_num)
            anchors_vaild_map: torch.Tensor(height, width, anchor_num)
            
        Returns:
            Dict[bbox_num, (y1, x1, y2, x2, class socre)]
        """
        
        self.eval()
        assert image_data.shape[0] == 1
        
        # Forward inference
        proposals, classes, regressions = self(
            image_data=image_data,
            anchors=anchors,
            anchors_vaild_map=anchors_vaild_map
        )
        proposals=proposals.cpu().numpy()
        classes=classes.cpu().numpy()
        regressions=regressions.cpu().numpy()
        
        # Convert proposal bboxes to the form of center point and size
        proposal_anchors = np.empty(proposals.shape)
        proposal_anchors[:, 0] = 0.5 * (proposals[:, 0] + proposals[:, 2])
        proposal_anchors[:, 1] = 0.5 * (proposals[:, 1] + proposals[:, 3])
        proposal_anchors[:, 2: 4] = proposals[:, 2: 4] - proposals[:, 0: 2]
        
        # Separate out results per class
        bboxes_and_scores_per_class = {}
        for class_idx in range(1, classes.shape[1]):
            # Get the box regression corresponding to this class
            bbox_regression_idx = (class_idx - 1) * 4
            bbox_regression = regressions[:, (bbox_regression_idx + 0): (bbox_regression_idx + 4)]
            bboxes_per_class = math_utils.compute_bbox(
                box_deltas=bbox_regression,
                anchors=proposal_anchors,
                box_delta_means=self._detector_bbox_delta_means,
                box_delta_stds=self._detector_bbox_delta_stds
            )
            
            # Clip to image boundaries
            bboxes_per_class[:, 0: : 2] = np.clip(bboxes_per_class[:, 0: : 2], 0, image_data.shape[2] - 1)
            bboxes_per_class[:, 1: : 2] = np.clip(bboxes_per_class[:, 1: : 2], 0, image_data.shape[3] - 1)
            
            # Get the score corresponding to this class
            scores_per_class = classes[:, class_idx]
            
            # Filter out whose socres are higher than score_threshold
            higher_score_idxs = np.where(scores_per_class > score_threshold)[0]
            bboxes_per_class = bboxes_per_class[higher_score_idxs]
            scores_per_class = scores_per_class[higher_score_idxs]
            bboxes_and_scores_per_class[class_idx] = (bboxes_per_class, scores_per_class)
            
        # Perform NMS per class
        scored_bboxes_per_class = {}
        for class_idx, (bboxes, scores) in bboxes_and_scores_per_class.items():
            idxs = nms(
                boxes=torch.from_numpy(bboxes).cuda(),
                scores=torch.from_numpy(scores).cuda(),
                iou_threshold=0.3
            ).cpu().numpy()
            bboxes=bboxes[idxs]
            scores=np.expand_dims(scores[idxs], axis=0)
            scored_bboxes = np.hstack([bboxes, scores.T])
            scored_bboxes_per_class[class_idx] =scored_bboxes
            
        return scored_bboxes_per_class
    
    def train_step(self, image_data, optimizer, anchors, anchors_vaild_map, gt_rpn_map, gt_rpn_object_indices, gt_rpn_background_indices, gt_bboxes):
        """
        One training step on a sample of data

        Args:
            image_data: torch.Tensor(batch_size, channels, height, width)
            optimizer: torch.optim.Optimizer
            anchors: torch.Tensor(height, width, [center_y, center_x, height, width] * anchor_num)
            anchors_vaild_map: torch.Tensor(height, width, anchor_num)
            gt_rpn_map: np.array(height, width, anchor_num, [trainable(1 for vaild an non-neutral anchors), symbol(1 for object anchor and 0 for the rest), dy, dx, dh, dw])
            gt_rpn_object_indices: List[np.array], indices (y, x, k) of all object anchors
            gt_rpn_background_indices: List[np.array], indices (y, x, k) of all background anchors
            gt_bboxes: List[List[dataset.training_sample.Bbox]], a list of ground truth object bboxes for each image in th batch
            
        Returns:
            loss: class and regression losses for both the RPN and the detector
        """
        
        self.train()
        
        # Clear accumulated gradient
        optimizer.zero_grad()
        
        # A bacth size of 1
        assert image_data.shape[0] == 1
        assert len(gt_rpn_map.shape) == 5 and gt_rpn_map.shape[0] == 1
        assert len(gt_rpn_object_indices) == 1
        assert len(gt_rpn_background_indices) == 1
        assert len(gt_bboxes) == 1
        
        image_shape = image_data.shape[1: ]
        
        # Extract features
        feature_map = self._feature_extractor(image_data)
        
        # Generate object proposals from RPN
        rpn_objectness_scores, rpn_bbox_regressions, proposals = self._rpn(
            feature_map,
            image_shape=image_shape,
            anchors=anchors,
            anchors_vaild_map=anchors_vaild_map,
            before_proposal_num=12000,
            after_proposal_num=2000
        )
        
        # Sample random mini-batch of anchors for RPN training
        gt_rpn_minibatch_map = self._sample_rpn_minibatch(
            rpn_map=gt_rpn_map,
            object_indices=gt_rpn_object_indices,
            background_indices=gt_rpn_background_indices
        )
        
        # Assign labels to proposals and take random sample for detector training
        proposals, gt_classes, gt_bbox_regressions = self._label_proposals(
            proposals=proposals,
            gt_bboxes = gt_bboxes[0],
            min_object_iou_threshold=0.5,
            min_background_iou_threshold=0.0
        )
        proposals, gt_classes, gt_bbox_regressions = self._sample_proposals(
            proposals=proposals,
            gt_classes=gt_classes,
            gt_bbox_regressions=gt_bbox_regressions,
            max_proposals=self._proposal_batch_size,
            positive_fraction=0.25
        )
        
        #Make sure RoI proposals and ground truths are detached from computational graph so that gradients are not propagated through them.
        proposals = proposals.detach()
        gt_classes = gt_classes.detach()
        gt_bbox_regressions = gt_bbox_regressions.detach()
        
        # Detect
        detector_classes, detector_bbox_regressions = self._detector(feature_map, proposals)
        
        # Compute losses
        rpn_class_loss, rpn_bbox_loss, detector_class_loss, detector_bbox_loss = loss.compute_losses(
            rpn_objectness_scores=rpn_objectness_scores,
            rpn_bbox_regressions=rpn_bbox_regressions,
            rpn_gt_map=gt_rpn_minibatch_map,
            detector_classes=detector_classes,
            detector_gt_classes=gt_classes,
            detector_bbox_regressions=detector_bbox_regressions,
            detector_gt_map=gt_bbox_regressions
        )
        total_loss = rpn_class_loss + rpn_bbox_loss + detector_class_loss + detector_bbox_loss
        Loss = FasterRCNN.Loss(
            rpn_class = rpn_class_loss.detach().cpu().item(),
            rpn_bbox = rpn_bbox_loss.detach().cpu().item(),
            detector_class = detector_class_loss.detach().cpu().item(),
            detector_bbox = detector_bbox_loss.detach().cpu().item(),
            total = total_loss.detach().cpu().item()
        )
        
        # Backprop
        total_loss.backward()
        
        # Optimize
        optimizer.step()
        
        return Loss
        
    def _sample_rpn_minibatch(self, rpn_map, object_indices, background_indices):
        """
        Selects anchors for training and produces a copy of the RPN ground truth map with only those anchors marked as trainable

        Args:
            rpn_map: np.ndarray(height, width, anchor_num, [trainable(1 for vaild an non-neutral anchors), symbol(1 for object anchor and 0 for the rest), dy, dx, dh, dw])
            object_indices: List[np.ndarray](indices (y, x, k) of all object anchors)
            background_indices: List[np.ndarray](indices (y, x, k) of all background anchors)

        Returns:
            rpn_minibatch_map: a copy of the RPN ground truth map with index 0 of the last dimension recomputed to include only anchors in the minibatch
        """
        # Only one batch
        assert rpn_map.shape[0] == 1
        assert len(object_indices) == 1
        assert len(background_indices) == 1
        
        positive_anchors = object_indices[0]
        negative_anchors = background_indices[0]
        assert len(positive_anchors) + len(negative_anchors) >= self._rpn_minibatch_size, "Image has insufficient anchors for RPN minibatch size of %d" % self._rpn_minibatch_size
        assert len(positive_anchors) > 0, "Image does not have any positive anchors"
        assert self._rpn_minibatch_size % 2 == 0, "RPN minibatch size must be evenly divisible"

        # Sample, producing indices into the index maps
        positive_anchor_num = len(positive_anchors)
        negative_anchor_num = len(negative_anchors)
        positive_sample_num = min(self._rpn_minibatch_size // 2, positive_anchor_num) # up to half the samples should be positive, if possible
        negative_sample_num = self._rpn_minibatch_size - positive_sample_num          # the rest should be negative
        positive_anchor_idxs = random.sample(range(positive_anchor_num), positive_sample_num)
        negative_anchor_idxs = random.sample(range(negative_anchor_num), negative_sample_num)

        # Construct index expressions into RPN map
        positive_anchors = positive_anchors[positive_anchor_idxs]
        negative_anchors = negative_anchors[negative_anchor_idxs]
        trainable_anchors = np.concatenate([positive_anchors, negative_anchors])
        batch_idxs = np.zeros(len(trainable_anchors))
        trainable_idxs = (batch_idxs, trainable_anchors[:, 0], trainable_anchors[:, 1], trainable_anchors[:, 2], 0)

        # Create a copy of the RPN map with samples set as trainable
        rpn_minibatch_map = rpn_map.clone()
        rpn_minibatch_map[:, :, :, :, 0] = 0
        rpn_minibatch_map[trainable_idxs] = 1

        return rpn_minibatch_map

    def _label_proposals(self, proposals, gt_bboxes, min_object_iou_threshold, min_background_iou_threshold):
        """
        determine which proposals generated by the RPN stage overlap with ground truth bboxes
        create ground truth labels for the subsequent detector stage

        Args:
            proposals: torch.Tensor(proposal_num, [y1, x1, y2, x2] * proposal_num) from RPN
            gt_bboxes: List[dataset.voc.Box], ground truth object bboxes
            min_background_iou_threshold: float, minimum IoU threshold with ground truth bboxes below which proposals are ignored entirely, proposals with an IoU threshold in the range [min_background_iou_threshold, min_object_iou_threshold) are labeled as background
            min_object_iou_threshold: float, minimum IoU threshold for a proposal to be labeled as an object.

        Returns:
            proposals: torch.Tensor(proposal_num, [y1, x1, y2, x2] * proposal_num) labeld as either objects or background
            classes: torch.Tensor(proposal_num, class_num)
            regressions: torch.Tensor(proposal_num, mask, [dy, dx, dh, dw] * (class_num - 1)), for each proposal assigned a non-background class, there will be 4 consecutive elements marked with 1 indicating the corresponding box delta target values are to be used
        """
        
        # Object threshold must be greater than background threshold
        assert min_background_iou_threshold < min_object_iou_threshold

        # Convert ground truth box corners to (M,4) tensor and class indices to (M,)
        gt_bbox_corners = np.array([box.corners for box in gt_bboxes], dtype=np.float32)
        gt_bbox_corners = torch.from_numpy(gt_bbox_corners).cuda()
        gt_bbox_class_id = torch.tensor([box.class_id for box in gt_bboxes], dtype=torch.long, device="cuda")

        # Let's be crafty and create some fake proposals that match the ground
        # truth boxes exactly. This isn't strictly necessary and the model should
        # work without it but it will help training and will ensure that there are
        # always some positive examples to train on.
        proposals = torch.vstack([proposals, gt_bbox_corners])

        # Compute IoU between each proposal (N,4) and each ground truth box (M,4)
        # -> (N, M)
        ious = math_utils.intersection_over_union_torch(boxes1=proposals, boxes2=gt_bbox_corners)

        # Find the best IoU for each proposal, the class of the ground truth box
        # associated with it, and the box corners
        best_ious = torch.max(ious, dim=1).values         # (N,) of maximum IoUs for each of the N proposals
        box_idxs = torch.argmax(ious, dim=1)              # (N,) of ground truth box index for each proposal
        gt_bbox_class_id = gt_bbox_class_id[box_idxs] # (N,) of class indices of highest-IoU box for each proposal
        gt_bbox_corners = gt_bbox_corners[box_idxs]       # (N,4) of box corners of highest-IoU box for each proposal

        # Remove all proposals whose best IoU is less than the minimum threshold
        # for a negative (background) sample. We also check for IoUs > 0 because
        # due to earlier clipping, we may get invalid 0-area proposals.
        idxs = torch.where((best_ious >= min_background_iou_threshold))[0]  # keep proposals w/ sufficiently high IoU
        proposals = proposals[idxs]
        best_ious = best_ious[idxs]
        gt_bbox_class_id = gt_bbox_class_id[idxs]
        gt_bbox_corners = gt_bbox_corners[idxs]

        # IoUs less than min_object_iou_threshold will be labeled as background
        gt_bbox_class_id[best_ious < min_object_iou_threshold] = 0

        # One-hot encode class labels
        proposal_num = proposals.shape[0]
        gt_classes = torch.zeros((proposal_num, self._class_num), dtype=torch.float32, device="cuda")  # (N,num_classes)
        gt_classes[torch.arange(proposal_num), gt_bbox_class_id] = 1.0

        # Convert proposals and ground truth boxes into "anchor" format (center
        # points and side lengths). For the detector stage, the proposals serve as
        # the anchors relative to which the final box predictions will be
        # regressed.
        proposal_centers = 0.5 * (proposals[:, 0: 2] + proposals[:, 2: 4])          # center_y, center_x
        proposal_sides = proposals[:, 2: 4] - proposals[:, 0: 2]                    # height, width
        gt_bbox_centers = 0.5 * (gt_bbox_corners[:, 0: 2] + gt_bbox_corners[:, 2: 4])  # center_y, center_x
        gt_bbox_sides = gt_bbox_corners[:, 2: 4] - gt_bbox_corners[:, 0: 2]            # height, width

        # Compute box delta regression targets (ty, tx, th, tw) for each proposal
        # based on the best box selected
        box_delta_targets = torch.empty((proposal_num, 4), dtype=torch.float32, device="cuda") # (N,4)
        box_delta_targets[:, 0: 2] = (gt_bbox_centers - proposal_centers) / proposal_sides # ty = (gt_center_y - proposal_center_y) / proposal_height, tx = (gt_center_x - proposal_center_x) / proposal_width
        box_delta_targets[:, 2: 4] = torch.log(gt_bbox_sides / proposal_sides)                 # th = log(gt_height / proposal_height), tw = (gt_width / proposal_width)
        box_delta_means = torch.tensor(self._detector_bbox_delta_means, dtype=torch.float32, device="cuda")
        box_delta_stds = torch.tensor(self._detector_bbox_delta_stds, dtype=torch.float32, device="cuda")
        box_delta_targets[:, : ] -= box_delta_means                               # mean adjustment
        box_delta_targets[:, : ] /= box_delta_stds                                # standard deviation scaling

        # Convert regression targets into a map of shape (N,2,4*(C-1)) where C is
        # the number of classes and [:,0,:] specifies a mask for the corresponding
        # target components at [:,1,:]. Targets are ordered (ty, tx, th, tw).
        # Background class 0 is not present at all.
        gt_bbox_regressions = torch.zeros((proposal_num, 2, 4 * (self._class_num - 1)), dtype=torch.float32, device="cuda")
        gt_bbox_regressions[:, 0, : ] = torch.repeat_interleave(gt_classes, repeats=4, dim=1)[:, 4: ]  # create masks using interleaved repetition, remembering to ignore class 0
        gt_bbox_regressions[:, 1, : ] = torch.tile(box_delta_targets, dims=(1, self._class_num - 1)) # populate regression targets with straightforward repetition (only those columns corresponding to class are masked on)

        return proposals, gt_classes, gt_bbox_regressions

    def _sample_proposals(self, proposals, gt_classes, gt_bbox_regressions, max_proposals, positive_fraction):
        if max_proposals <= 0:
            return proposals, gt_classes, gt_bbox_regressions

        # Get positive and negative (background) proposals
        class_indices = torch.argmax(gt_classes, axis = 1)  # (N,num_classes) -> (N,), where each element is the class index (highest score from its row)
        positive_indices = torch.where(class_indices > 0)[0]
        negative_indices = torch.where(class_indices <= 0)[0]
        positive_proposal_num = len(positive_indices)
        negative_proposal_num = len(negative_indices)

        # Select positive and negative samples, if there are enough. Note that the
        # number of positive samples can be either the positive fraction of the
        # *actual* number of proposals *or* the *desired* number (max_proposals).
        # In practice, these yield virtually identical results but the latter
        # method will yield slightly more positive samples in the rare cases when
        # the number of proposals is below the desired number. Here, we use the
        # former method but others, such as Yun Chen, use the latter. To implement
        # it, replace sample_num with max_proposals in the line that computes
        # positive_sample_num. I am not sure what the original Faster R-CNN
        # implementation does.
        sample_num = min(max_proposals, len(class_indices))
        positive_sample_num = min(round(sample_num * positive_fraction), positive_proposal_num)
        negative_sample_num = min(sample_num - positive_sample_num, negative_proposal_num)

        # Do we have enough?
        if positive_sample_num <= 0 or negative_sample_num <= 0:
            return proposals[[]], gt_classes[[]], gt_bbox_regressions[[]] # return 0-length tensors

        # Sample randomly
        positive_sample_indices = positive_indices[torch.randperm(len(positive_indices))[0:positive_sample_num]]
        negative_sample_indices = negative_indices[torch.randperm(len(negative_indices))[0:negative_sample_num]]
        indices = torch.cat([positive_sample_indices, negative_sample_indices])

        # Return
        return proposals[indices], gt_classes[indices], gt_bbox_regressions[indices]