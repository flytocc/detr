from . import _C


def poly_iou(boxes1, boxes2):
    assert boxes1.ndim == 2 and boxes1.size(-1) == 8
    assert boxes2.ndim == 2 and boxes2.size(-1) == 8

    return _C.poly_iou(boxes1, boxes2)
