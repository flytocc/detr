# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import math

import torch
from torchvision.ops.boxes import box_area

from .poly_iou import poly_iou


def add_angle(boxes, zero_angle=False):
    assert boxes.size(-1) == 4
    if zero_angle:
        angle = boxes.new_zeros(len(boxes), 1)
    else:
        ox, oy = (boxes[..., :2] - 0.5).split(1, dim=-1)
        angle = torch.atan(ox / -oy) * (180 / math.pi)
    return torch.cat((boxes, angle), dim=-1)


def rbbox_to_poly(rbboxes):
    xc, yc, w, h, alpha = rbboxes.unbind(dim=-1)
    p1 = [- w / 2, - h / 2]
    p2 = [  w / 2, - h / 2]
    p3 = [  w / 2,   h / 2]
    p4 = [- w / 2,   h / 2]

    alpha_rad = -alpha * (math.pi / 180)
    cos_theta, sin_theta = torch.cos(alpha_rad), torch.sin(alpha_rad)

    rp1 = [p1[0] * cos_theta + p1[1] * sin_theta + xc,
           p1[1] * cos_theta - p1[0] * sin_theta + yc]
    rp2 = [p2[0] * cos_theta + p2[1] * sin_theta + xc,
           p2[1] * cos_theta - p2[0] * sin_theta + yc]
    rp3 = [p3[0] * cos_theta + p3[1] * sin_theta + xc,
           p3[1] * cos_theta - p3[0] * sin_theta + yc]
    rp4 = [p4[0] * cos_theta + p4[1] * sin_theta + xc,
           p4[1] * cos_theta - p4[0] * sin_theta + yc]
    poly = rp1 + rp2 + rp3 + rp4

    return torch.stack(poly, dim=-1)


def generalized_rbbox_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [xc, yc, w, h, a] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    N, M = len(boxes1), len(boxes2)
    if N * M == 0:
        return boxes1.new_zeros(N, M)

    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]

    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:4] >= 0).all()
    assert (boxes2[:, 2:4] >= 0).all()

    poly1 = rbbox_to_poly(boxes1)
    poly2 = rbbox_to_poly(boxes2)

    iou = poly_iou(poly1, poly2)
    union = (area1[:, None] + area2[None]) / (iou + 1)

    poly1 = poly1.view(-1, 4, 2)
    poly2 = poly2.view(-1, 4, 2)
    lt = torch.min(poly1.min(dim=1)[0][:, None], poly2.min(dim=1)[0][None])
    rb = torch.max(poly1.max(dim=1)[0][:, None], poly2.max(dim=1)[0][None])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
