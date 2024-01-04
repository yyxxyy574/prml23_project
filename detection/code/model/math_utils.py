# Common math utility functions

import torch
import numpy as np 

def intersection_over_union(boxes1, boxes2):
    """
    Computes intersection-over-union (IoU) for multiple boxes in parallel.
    
    Args:
        boxes1, boxes2: np.ndarray, box corners, shaped (N, 4), with each box as (y1, x1, y2, x2)

    Returns:
        np.ndarray, IoUs for each pair of boxes in boxes1 and boxes2
    """
    
    top_left_point = np.maximum(boxes1[:, None, 0: 2], boxes2[:, 0: 2])                               
    bottom_right_point = np.minimum(boxes1[:, None, 2: 4], boxes2[:, 2: 4])                             
    well_ordered_mask = np.all(top_left_point<bottom_right_point, axis=2) # whether top_left_x < bottom_right_x and top_left_y < bottom_right_y (meaning boxes may intersect)
    intersection_areas = well_ordered_mask * np.prod(bottom_right_point-top_left_point, axis=2) # (intersection area (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
    areas1 = np.prod(boxes1[:, 2: 4] - boxes1[:, 0: 2], axis=1) # areas of boxes1
    areas2 = np.prod(boxes2[:, 2: 4] - boxes2[:, 0: 2], axis=1) # areas of boxes2
    union_areas = areas1[:, None] + areas2 - intersection_areas # union areas of both boxes
    epsilon = 1e-7
    return intersection_areas / (union_areas + epsilon)

def intersection_over_union_torch(boxes1, boxes2):
    """
    Computes intersection-over-union (IoU) for multiple boxes in parallel on pytorch Tensor.
    
    Args:
        boxes1, boxes2: torch.Tensor, box corners, shaped (N, 4), with each box as (y1, x1, y2, x2)

    Returns:
        torch.Tensor, IoUs for each pair of boxes in boxes1 and boxes2
    """
    top_left_point = torch.maximum(boxes1[:, None, 0: 2], boxes2[:, 0: 2])                                 
    bottom_right_point = torch.minimum(boxes1[:, None, 2: 4], boxes2[:, 2: 4])                             
    well_ordered_mask = torch.all(top_left_point<bottom_right_point, axis=2) # whether top_left_x < bottom_right_x and top_left_y < bottom_right_y (meaning boxes may intersect)
    intersection_areas = well_ordered_mask * torch.prod(bottom_right_point-top_left_point, dim=2) # (intersection area (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
    areas1 = torch.prod(boxes1[: ,2: 4] - boxes1[:, 0: 2], dim=1) # areas of boxes1
    areas2 = torch.prod(boxes2[:, 2: 4] - boxes2[:, 0: 2], dim=1) # areas of boxes2
    union_areas = areas1[:, None] + areas2 - intersection_areas # union areas of both boxes
    epsilon = 1e-7
    return intersection_areas / (union_areas + epsilon)

def compute_bbox(box_deltas, anchors, box_delta_means, box_delta_stds):
    """
    Compute bbox according to base anchor (e.g., RPN anchors or proposals) and box deltas ((ty, tx, th, tw) as described by the Fast R-CNN and Faster R-CNN papers)

    Args:
        box_deltas: np.ndarray, (batch_size, height, width, deltas to be applied to anchors[dy, dx, dh, dw] * anchor_num)
        anchors: np.ndarray, (height, width, [center_y, center_x, height, width] * anchor_num)
        box_delta_means: np.ndarray, mean ajustment to box deltas, (4,), to be added after standard deviation scaling and before conversion to actual box coordinates
        box_delta_stds: np.ndarray, standard deviation adjustment to box deltas, (4,). box deltas are first multiplied by these values

    Returns:
        boxes: np.ndarray, (corresponding box corners[y1, x1, y2, x2] * anchor_num)
    """
    box_deltas = box_deltas * box_delta_stds + box_delta_means
    center = anchors[:, 2: 4] * box_deltas[:, 0: 2] + anchors[:, 0: 2] # center_x = anchor_width * dx + anchor_center_x, center_y = anchor_height * dy + anchor_center_y
    size = anchors[:, 2: 4] * np.exp(box_deltas[:, 2: 4]) # width = anchor_width * exp(dw), height = anchor_height * exp(dh)
    boxes = np.empty(box_deltas.shape)
    boxes[:, 0: 2] = center - 0.5 * size # y1, x1
    boxes[:, 2: 4] = center + 0.5 * size # y2, x2
    return boxes

def compute_bbox_torch(box_deltas, anchors, box_delta_means, box_delta_stds):
    """
    Compute bbox according to base anchor (e.g., RPN anchors or proposals) and box deltas ((ty, tx, th, tw) as described by the Fast R-CNN and Faster R-CNN papers), on pytorch Tensor

    Args:
        box_deltas: torch.Tensor, (batch_size, height, width, deltas to be applied to anchors[dy, dx, dh, dw] * anchor_num)
        anchors: torch.Tensor, (height, width, [center_y, center_x, height, width] * anchor_num)
        box_delta_means: torch.Tensor, mean ajustment to box deltas, (4,), to be added after standard deviation scaling and before conversion to actual box coordinates
        box_delta_stds: torch.Tensor, standard deviation adjustment to box deltas, (4,). box deltas are first multiplied by these values

    Returns:
        boxes: torch.Tensor, (corresponding box corners[y1, x1, y2, x2] * anchor_num)
    """
    box_deltas = box_deltas * box_delta_stds + box_delta_means
    center = anchors[:,2:4] * box_deltas[:,0:2] + anchors[:,0:2]#  center_x = anchor_width * dx + anchor_center_x, center_y = anchor_height * dy + anchor_center_y
    size = anchors[:,2:4] * torch.exp(box_deltas[:,2:4]) # width = anchor_width * exp(dw), height = anchor_height * exp(dh)
    boxes = torch.empty(box_deltas.shape, dtype = torch.float32, device = "cuda")
    boxes[:,0:2] = center - 0.5 * size # y1, x1
    boxes[:,2:4] = center + 0.5 * size # y2, x2
    return boxes