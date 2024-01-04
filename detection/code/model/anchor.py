# Generate anchors and related information, whether the anchor should be included in training, and the box regression targets, for RPN

import numpy as np
from math import sqrt
import itertools

from . import math_utils

def compute_anchor_size():
    areas = [128 * 128, 256 * 256, 512 * 512] # pixels
    rations = [0.5, 1.0, 2.0] # x:1

    # Generate anchor sizes from all 9 combinations of area and ration
    heights = np.array([sqrt(areas[i] / rations[j]) * rations[j] for (i, j) in itertools.product(range(3), range(3))])
    widths = np.array([sqrt(areas[i] / rations[j]) for (i, j) in itertools.product(range(3), range(3))])

    return np.vstack([heights, widths]).T

def generate_anchors(image_shape, feature_map_shape, feature_pixels):
    """
    Given image_shape, feature_map_shape, and feature_pixels, generate 9 different anchors for each feature map cell

    Args:
        image_shape: Tuple[channels(int), height(int), width(int)]
        feature_map_shape: Tuple[channels(int), height(int), width(int)]
        feature_pixels: int, distance in pixels between anchors

    Returns:
        anchors: np.array(height, width, [center_y, center_x, height, width] * anchor_num)
        anchors_vaild_map: np.array(height, width, anchor_num)
    """

    assert len(image_shape) == 3

    # Each anchor is specified by corners (y1, x1, y2,x2)
    anchor_sizes = compute_anchor_size()
    anchor_num = anchor_sizes.shape[0]
    anchors = np.empty((anchor_num, 4))
    anchors[:, 0: 2] = -0.5 * anchor_sizes # top-left, y1, x1
    anchors[:, 2: 4] = +0.5 * anchor_sizes # bottom-right y2, x2

    # Feature map shape
    height = feature_map_shape[-2]
    width = feature_map_shape[-1]

    # Map of coordinates
    y_cell_coords = np.arange(height)
    x_cell_coords = np.arange(width)
    cell_coords = np.array(np.meshgrid(y_cell_coords, x_cell_coords)).transpose([2, 1, 0])
    # Convert coordinates to image pixels at center of each cell
    center_points = cell_coords *  feature_pixels + 0.5 * feature_pixels
    center_points = np.tile(center_points, reps=2)
    center_points = np.tile(center_points, reps=anchor_num)

    # Add anchors to each cell
    anchors = center_points.astype(np.float32) + anchors.flatten()
    anchors = anchors.reshape((height * width * anchor_num), 4) # (height, width, 4 * anchor_num) -> (height * width * anchor_num, 4)
    
    # Get vaild anchors that do not cross image boundaries
    image_height, image_width = image_shape[1: ]
    anchors_vaild_map = np.all((anchors[:, 0: 2] >= [0, 0]) & (anchors[:, 2: 4] <= [image_height, image_width]), axis=1)
    anchors_vaild_map = anchors_vaild_map.reshape((height, width, anchor_num))

    # Change anchor forms
    anchors_copy = anchors
    anchors = np.empty((anchors.shape[0], 4))
    anchors[:, 0: 2] = 0.5 * (anchors_copy[:, 0: 2] + anchors_copy[:, 2: 4])
    anchors[:, 2: 4] = anchors_copy[:, 2: 4] - anchors_copy[:, 0: 2]
    anchors = anchors.reshape((height, width, anchor_num * 4))

    return anchors.astype(np.float32), anchors_vaild_map.astype(np.float32)

def generate_rpn_map(anchors, anchors_vaild_map, gt_bboxes, object_iou_threshold=0.7, background_iou_threshold=0.3):
    """
    Generate map containing ground truth data for training the region proposal network

    Args:
        anchors: np.array(height, width, [center_y, center_x, height, width] * anchor_num)
        anchors_vaild_map: np.array(height, width, anchor_num)
        gt_bboxes: list of ground truth bboxes
        object_iou_threshold: float, iou between an anchor and a ground truth bbox above which will be labeld as an object (positive) anchor
        background_iou_threshold: float, iou between an anchor and a ground truth bbox below which will be labeld as background (negative)

    Returns:
        rpn_map: np.array(height, width, anchor_num, [trainable(1 for vaild an non-neutral anchors), symbol(1 for object anchor and 0 for the rest), dy, dx, dh, dw])
        object_anchor_idxs: List[np.array], indices (y, x, k) of all object anchors
        background_anchor_idxs: List[np.array], indices (y, x, k) of all background anchors
    """

    height, width, anchor_num = anchors_vaild_map.shape

    # Convert ground truth box corners to (M,4) tensor and class indices to (M,)
    gt_bbox_corners = np.array([box.corners for box in gt_bboxes])
    gt_bbox_num = len(gt_bboxes)

    # Compute ground truth box center points and side lengths
    gt_bbox_centers = 0.5 * (gt_bbox_corners[:, 0: 2] + gt_bbox_corners[:, 2: 4])
    gt_bbox_sides = gt_bbox_corners[:, 2: 4] - gt_bbox_corners[:, 0: 2]

    # Flatten anchor boxes to (N,4) and convert to corners
    anchors_copy = anchors.reshape((-1, 4))
    anchors = np.empty(anchors_copy.shape)
    anchors[:, 0: 2] = anchors_copy[:, 0: 2] - 0.5 * anchors_copy[:, 2: 4]  # y1, x1
    anchors[:, 2: 4] = anchors_copy[:, 0: 2] + 0.5 * anchors_copy[:, 2: 4]  # y2, x2
    n = anchors.shape[0]

    # Initialize all anchors initially as negative (background). We will also
    # track which ground truth box was assigned to each anchor.
    objectness_score = np.full(n, -1) # RPN class: 0 = background, 1 = foreground, -1 = ignore (these will be marked as invalid in the truth map)
    gt_bbox_assignments = np.full(n, -1) # -1 means no box

    # Compute IoU between each anchor and each ground truth box, (N,M).
    ious = math_utils.intersection_over_union(boxes1=anchors, boxes2=gt_bbox_corners)

    # Need to remove anchors that are invalid (straddle image boundaries) from
    # consideration entirely and the easiest way to do this is to wipe out their
    # IoU scores
    ious[anchors_vaild_map.flatten() == 0, : ] = -1.0

    # Find the best IoU ground truth box for each anchor and the best IoU anchor
    # for each ground truth box.
    #
    # Note that ious == max_iou_per_gt_bbox tests each of the N rows of ious
    # against the M elements of max_iou_per_gt_bbox, column-wise. np.where() then
    # returns all (y,x) indices of matches as a tuple: (y_indices, x_indices).
    # The y indices correspond to the N dimension and therefore indicate anchors
    # and the x indices correspond to the M dimension (ground truth boxes).
    max_iou_per_anchor = np.max(ious, axis=1) # (N,)
    best_box_idx_per_anchor = np.argmax(ious, axis=1) # (N,)
    max_iou_per_gt_bbox = np.max(ious, axis=0) # (M,)
    highest_iou_anchor_idxs = np.where(ious == max_iou_per_gt_bbox)[0] # get (L,) indices of anchors that are the highest-overlapping anchors for at least one of the M boxes

    # Anchors below the minimum threshold are negative
    objectness_score[max_iou_per_anchor < background_iou_threshold] = 0

    # Anchors that meet the threshold IoU are positive
    objectness_score[max_iou_per_anchor >= object_iou_threshold] = 1

    # Anchors that overlap the most with ground truth boxes are positive
    objectness_score[highest_iou_anchor_idxs] = 1

    # We assign the highest IoU ground truth box to each anchor. If no box met
    # the IoU threshold, the highest IoU box may happen to be a box for which
    # the anchor had the highest IoU. If not, then the objectness score will be
    # negative and the box regression won't ever be used.
    gt_bbox_assignments[:] = best_box_idx_per_anchor

    # Anchors that are to be ignored will be marked invalid. Generate a mask to
    # multiply anchors_vaild_map by (-1 -> 0, 0 or 1 -> 1). Then mark ignored
    # anchors as 0 in objectness score because the score can only really be 0 or
    # 1.
    enable_mask = (objectness_score >= 0).astype(np.float32)
    objectness_score[objectness_score < 0] = 0

    # Compute box delta regression targets for each anchor
    box_delta_targets = np.empty((n, 4))
    box_delta_targets[:, 0: 2] = (gt_bbox_centers[gt_bbox_assignments] - anchors_copy[:, 0: 2]) / anchors_copy[:, 2: 4] # ty = (box_center_y - anchor_center_y) / anchor_height, tx = (box_center_x - anchor_center_x) / anchor_width
    box_delta_targets[:, 2: 4] = np.log(gt_bbox_sides[gt_bbox_assignments] / anchors_copy[:, 2: 4]) # th = log(box_height / anchor_height), tw = log(box_width / anchor_width)

    # Assemble RPN ground truth map
    rpn_map = np.zeros((height, width, anchor_num, 6))
    rpn_map[:, : ,:, 0] = anchors_vaild_map * enable_mask.reshape((height, width, anchor_num)) # trainable anchors (object or background; excludes boundary-crossing invalid and neutral anchors)
    rpn_map[:, :, :, 1] = objectness_score.reshape((height, width, anchor_num))
    rpn_map[:, :, :, 2: 6] = box_delta_targets.reshape((height, width,anchor_num, 4))

    # Return map along with positive and negative anchors
    rpn_map_coords = np.transpose(np.mgrid[0:height,0:width,0:anchor_num], (1,2,3,0)) # shape (height,width,k,3): every index (y,x,k,:) returns its own coordinate (y,x,k)
    object_anchor_idxs = rpn_map_coords[np.where((rpn_map[:, :, :, 1] > 0) & (rpn_map[:, :, :, 0] > 0))] # shape (N,3), where each row is the coordinate (y,x,k) of a positive sample
    background_anchor_idxs = rpn_map_coords[np.where((rpn_map[:, :, :, 1] == 0) & (rpn_map[:, :, :, 0] > 0))] # shape (N,3), where each row is the coordinate (y,x,k) of a negative sample
    return rpn_map.astype(np.float32), object_anchor_idxs, background_anchor_idxs