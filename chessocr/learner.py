# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02-learner.ipynb (unless otherwise specified).

__all__ = ['NoLabelBBoxLabeler', 'BBoxTruth', 'iou', 'NoLabelBBoxBlock']

# Cell
from fastcore.transform import *
from fastai.data.all import *
from fastai.vision.all import *
from fastai.test_utils import synth_learner
import pandas as pd
from pathlib import Path
from nbdev import show_doc
import os
import chessocr

# Cell
class NoLabelBBoxLabeler(Transform):
    """ Bounding box labeler with no label """
    def setups(self, x): noop
    def decode (self, x, **kwargs):
        self.bbox,self.lbls = None,None
        return self._call('decodes', x, **kwargs)

    def decodes(self, x:TensorBBox):
        self.bbox = x
        return self.bbox if self.lbls is None else LabeledBBox(self.bbox, self.lbls)

# Cell
class BBoxTruth:
    """ get bounding box location from DataFrame """
    def __init__(self, df): self.df=df

    def __call__(self, o):
        size,x,y,*_ =self.df.iloc[int(o.stem)-1]
        return [[x,y, x+size, y+size]]

# Cell
def iou(pred, target):
    """ Vectorized Intersection Over Union calculation """
    target = Tensor.cpu(target).squeeze(1)
    pred = Tensor.cpu(pred)
    ab = np.stack([pred, target])
    intersect_area = np.maximum(ab[:, :, [2, 3]].min(axis=0) - ab[:, :, [0, 1]].max(axis=0), 0).prod(axis=1)
    union_area = ((ab[:, :, 2] - ab[:, :, 0]) * (ab[:, :, 3] - ab[:, :, 1])).sum(axis=0) - intersect_area
    return (intersect_area / union_area).mean()

# Cell
NoLabelBBoxBlock = TransformBlock(type_tfms=TensorBBox.create,
                             item_tfms=[PointScaler, NoLabelBBoxLabeler])