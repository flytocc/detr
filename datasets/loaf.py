# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import math
from .coco import CocoDetection, make_coco_transforms


def poly_to_hbb(poly):
    return [min(poly[0::2]), min(poly[1::2]), max(poly[0::2]), max(poly[1::2])]


def rbbox_to_poly(rbbox):
    xc, yc, w, h, alpha = rbbox
    p1 = [- w / 2, - h / 2]
    p2 = [  w / 2, - h / 2]
    p3 = [  w / 2,   h / 2]
    p4 = [- w / 2,   h / 2]

    alpha_rad = - alpha / 180.
    cos_theta, sin_theta = math.cos(alpha_rad), math.sin(alpha_rad)

    rp1 = [p1[0] * cos_theta + p1[1] * sin_theta + xc,
           p1[1] * cos_theta - p1[0] * sin_theta + yc]
    rp2 = [p2[0] * cos_theta + p2[1] * sin_theta + xc,
           p2[1] * cos_theta - p2[0] * sin_theta + yc]
    rp3 = [p3[0] * cos_theta + p3[1] * sin_theta + xc,
           p3[1] * cos_theta - p3[0] * sin_theta + yc]
    rp4 = [p4[0] * cos_theta + p4[1] * sin_theta + xc,
           p4[1] * cos_theta - p4[0] * sin_theta + yc]
    poly = rp1 + rp2 + rp3 + rp4

    return poly


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "images/train", root / "annotations/original_resolution" / f'{mode}_train.json'),
        "val": (root / "images/val", root / "annotations/original_resolution" / f'{mode}_val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)

    for ann in dataset.coco.dataset['annotations']:
        x0, y0, x1, y1 = poly_to_hbb(rbbox_to_poly(ann['bbox']))
        ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]

    return dataset
