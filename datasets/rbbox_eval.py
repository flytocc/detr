import copy
import json
import math
import time

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from shapely.geometry import Polygon

import torch

from util.poly_iou import poly_iou


def loadRes(self, resFile):
    res = COCO()
    res.dataset['images'] = [img for img in self.dataset['images']]

    print('Loading and preparing results...')
    tic = time.time()
    if type(resFile) == str:
        with open(resFile) as f:
            anns = json.load(f)
    elif type(resFile) == np.ndarray:
        anns = self.loadNumpyAnnotations(resFile)
    else:
        anns = resFile
    assert type(anns) == list, 'results in not an array of objects'
    annsImgIds = [ann['image_id'] for ann in anns]
    assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
           'Results do not correspond to current coco set'
    if 'bbox' in anns[0] and not anns[0]['bbox'] == []:
        res.dataset['categories'] = copy.deepcopy(
            self.dataset['categories'])
        for id, ann in enumerate(anns):
            bb = ann['bbox']
            if 'segmentation' not in ann:
                ann['segmentation'] = rbbox2poly(bb)
            ann['area'] = bb[2] * bb[3]
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    print('DONE (t={:0.2f}s)'.format(time.time() - tic))

    res.dataset['annotations'] = anns
    res.createIndex()
    return res


class RBBoxCOCOeval(COCOeval):

    def __init__(self, *args, **kwargs):
        assert kwargs['iouType'] == 'bbox', 'iouType not supported'
        super().__init__(*args, **kwargs)

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        try:
            ious = self.ritbox_iou_cuda(d, g, iscrowd)
        except:
            ious = self.ritbox_iou(d, g, iscrowd)
        return ious

    def ritbox_iou_cuda(self, d, g, iscrowd):
        d = [rbbox2poly(rbbox) for rbbox in d]
        g = [rbbox2poly(rbbox) for rbbox in g]
        d = torch.as_tensor(d, dtype=torch.float32, device='cuda')
        g = torch.as_tensor(g, dtype=torch.float32, device='cuda')
        d = d.view(d.size(0), 8)
        g = g.view(g.size(0), 8)
        return poly_iou(d, g).cpu().numpy()

    def ritbox_iou(self, d, g, iscrowd):
        num_d = len(d)
        num_g = len(g)
        ious = np.zeros((num_d, num_g))
        for j in range(num_g):
            if iscrowd[j]:
                continue
            for i in range(num_d):
                d_poly = rbbox2poly(d[i])
                g_poly = rbbox2poly(g[j])
                quad_a = Polygon(np.array(d_poly).reshape(4, 2)).convex_hull
                quad_b = Polygon(np.array(g_poly).reshape(4, 2)).convex_hull

                if not quad_a.intersects(quad_b):
                    iou = 0
                else:
                    inter_area = quad_a.intersection(quad_b).area
                    area_a = quad_a.area
                    area_b = quad_b.area
                    iou = float(inter_area) / (area_a + area_b - inter_area)

                ious[i, j] = iou

        return ious


def rbbox2poly(rbbox):
    cx, cy, w, h, alpha = rbbox
    p1 = [- w / 2, - h / 2]
    p2 = [  w / 2, - h / 2]
    p3 = [  w / 2,   h / 2]
    p4 = [- w / 2,   h / 2]

    alpha_rad = -alpha * math.pi / 180
    cos_theta, sin_theta = math.cos(alpha_rad), math.sin(alpha_rad)

    rp1 = [p1[0] * cos_theta + p1[1] * sin_theta + cx,
           p1[1] * cos_theta - p1[0] * sin_theta + cy]
    rp2 = [p2[0] * cos_theta + p2[1] * sin_theta + cx,
           p2[1] * cos_theta - p2[0] * sin_theta + cy]
    rp3 = [p3[0] * cos_theta + p3[1] * sin_theta + cx,
           p3[1] * cos_theta - p3[0] * sin_theta + cy]
    rp4 = [p4[0] * cos_theta + p4[1] * sin_theta + cx,
           p4[1] * cos_theta - p4[0] * sin_theta + cy]
    poly = rp1 + rp2 + rp3 + rp4

    return poly
