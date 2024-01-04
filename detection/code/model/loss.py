# Loss functions

import torch
from torch.nn import functional as F

def compute_rpn_class_loss(rpn_objectness_scores, rpn_gt_map):
    """
    compute RPN class loss

    Args:
        rpn_objectness_scores: torch.Tensor(batch_size, height, width, anchor_num) containing objectness scores (background = 0, object = 1)
        rpn_gt_map: torch.Tensor(height, width, anchor_num, [trainable(1 for vaild an non-neutral anchors), symbol(1 for object anchor and 0 for the rest), dy, dx, dh, dw])
        
    Returns:
        scalar loss
    """

    epsilon = 1e-7
    
    # Convert rpn_gt_map into rpn_gt_scores formed as same as rpn_objectness_scores
    rpn_gt_scores = rpn_gt_map[:, :, :, :, 1].reshape(rpn_objectness_scores.shape)
    
    # Element-wise loss for all anchors
    rpn_class_loss = F.binary_cross_entropy(input=rpn_objectness_scores, target=rpn_gt_scores, reduction="none")
    
    # Anchors included in the mini-batch
    rpn_gt_included = rpn_gt_map[:, :, :, :, 0].reshape(rpn_objectness_scores.shape)
    
    # The number of anchors actually used in the mini-batch
    class_num = torch.count_nonzero(rpn_gt_included) + epsilon
    
    # Zero out the ones which should not have been included
    rpn_class_loss = rpn_gt_included * rpn_class_loss
    
    # Return the normalized total loss
    return torch.sum(rpn_class_loss) / class_num

def compute_rpn_regression_loss(rpn_bbox_regressions, rpn_gt_map):
    """
    compute RPN regression loss

    Args:
        rpn_bbox_regression: torch.Tensor(batch_size, height, width, deltas to be applied to anchors[dy, dx, dh, dw] * ahchor_num)
        rpn_gt_map: torch.Tensor(height, width, anchor_num, [trainable(1 for vaild an non-neutral anchors), symbol(1 for object anchor and 0 for the rest), dy, dx, dh, dw])
        
    Returns:
        scalar loss
    """
    
    epsilon = 1e-7
    scale_factor = 1.0
    sigma = 3.0
    sigma_squared = sigma * sigma

    # Convert rpn_gt_map into rpn_gt_regressions formed as same as rpn_bbox_regressions
    rpn_gt_regressions = rpn_gt_map[:, :, :, :, 2: 6].reshape(rpn_bbox_regressions.shape)
    
    # Element-wise loss for all anchors
    d = rpn_gt_regressions - rpn_bbox_regressions
    d_abs = torch.abs(d)
    is_negative_branch = (d_abs < (1.0 / sigma_squared)).float()
    R_negative_branch = 0.5 * d * d * sigma_squared
    R_positive_branch = d_abs - 0.5 / sigma_squared
    rpn_regression_loss = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch
    
    # Anchors included in the mini-batch
    rpn_gt_included = rpn_gt_map[:, :, :, :, 0].reshape(rpn_gt_map.shape[0: 4])
    # Anchors corresponding to obejcts
    rpn_gt_positive = rpn_gt_map[:, :, :, :, 1].reshape(rpn_gt_map.shape[0: 4])
    rpn_gt_used = rpn_gt_included * rpn_gt_positive
    rpn_gt_used = rpn_gt_used.repeat_interleave(repeats=4, dim=3)
    
    # The number of anchors actually used in the mini-batch
    regression_num = torch.count_nonzero(rpn_gt_included) + epsilon
    
    # Zero out the ones which should not have been included
    rpn_regression_loss = rpn_gt_used * rpn_regression_loss
    
    # Return the normalized total loss
    return scale_factor * torch.sum(rpn_regression_loss) / regression_num

def compute_detector_class_loss(detector_classes, detector_gt_classes):
    """
    Compute detector class loss
    
    Args:
        detector_classes: torch.Tensor(roi_num, class_num)
        detector_bbox_regressions: torch.Tensor(roi_num, [dy, dx, dh, dw] * (class_num - 1))
        
    Returns:
        scalar loss
    """
    
    epsilon = 1e-7
    scale_fator = 1.0
    
    detector_class_loss_per_row = -(detector_gt_classes * torch.log(detector_classes + epsilon)).sum(dim=1)
    num = detector_class_loss_per_row.shape[0] + epsilon
    
    return scale_fator * torch.sum(detector_class_loss_per_row) / num
        
        
def compute_detector_regression_loss(detector_bbox_regressions, detector_gt_map):
    """
    Compute detector regression loss

    Args:
        detector_bbox_regressions: torch.Tensor(roi_num, [dy, dx, dh, dw] * (class_num - 1))
        detector_gt_regressions: torch.Tensor(roi_num, 2, [dy, dx, dh, dw] * (class_num - 1))
        
    Returns:
        scalar loss
    """
    
    epsilon = 1e-7
    scale_factor = 1.0
    sigma = 1.0
    sigma_squared = sigma * sigma
    
    # Convert detector_gt_map into detector_gt_regressions formed as same as detector_bbox_regressions
    detector_gt_regressions = detector_gt_map[:, 1, : ]
        
    # Element-wise loss for all anchors
    d = detector_gt_regressions - detector_bbox_regressions
    d_abs = torch.abs(d)
    is_negative_branch = (d_abs < (1.0 / sigma_squared)).float()
    R_negative_branch = 0.5 * d * d * sigma_squared
    R_positive_branch = d_abs - 0.5 / sigma_squared
    detector_regression_loss = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch
    
    # ROIs corresponding to valid targets
    detector_gt_used = detector_gt_map[:, 0, : ]
    
    # The number of ROIs
    roi_num = detector_gt_regressions.shape[0] + epsilon
    
    # Zero out the ones which should not have been included
    detector_regression_loss = detector_gt_used * detector_regression_loss
    
    # Return the normalized total loss
    return scale_factor * torch.sum(detector_regression_loss) / roi_num

def compute_losses(rpn_objectness_scores, rpn_bbox_regressions, rpn_gt_map, detector_classes, detector_gt_classes, detector_bbox_regressions, detector_gt_map):
    
    rpn_class_loss = compute_rpn_class_loss(rpn_objectness_scores, rpn_gt_map)
    rpn_bbox_loss = compute_rpn_regression_loss(rpn_bbox_regressions, rpn_gt_map)
    detector_class_loss = compute_detector_class_loss(detector_classes, detector_gt_classes)
    detector_bbox_loss = compute_detector_regression_loss(detector_bbox_regressions, detector_gt_map)
    
    return rpn_class_loss, rpn_bbox_loss, detector_class_loss, detector_bbox_loss
    