# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
CrowdHuman dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

from .coco import CocoDetection, make_coco_transforms


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    PATHS = {
        "train": (root / "images", root / "annotations/pedestrain_only_train.json"),
        "val": (root / "images", root / "annotations/pedestrain_only_val.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
