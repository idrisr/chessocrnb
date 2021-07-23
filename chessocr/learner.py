# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03-learner.ipynb (unless otherwise specified).

__all__ = ['NoLabelBBoxLabeler', 'BBoxTruth', 'iou', 'NoLabelBBoxBlock']

# Cell
from fastai.data.all import *
from fastai.test_utils import synth_learner
from fastai.vision.all import *
from fastcore.transform import *
from pathlib import Path

import chessocr
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('pdf')

# Cell
class NoLabelBBoxLabeler(Transform):
    """ Bounding box labeler with no label """
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


def learn(dls):
    learn = cnn_learner(dls, resnet18, 
                        metrics=[iou], 
                        loss_func=MSELossFlat())
    learn.model.cuda()
    learn.fine_tune(10)
    learn.show_results()
    plt.savefig("results")


if __name__ == '__main__':
    data_url = Path.home()/".fastai/data/chess"
    df = pd.read_csv(data_url/'annotations.csv', index_col=0)
    block = DataBlock(
        blocks=(ImageBlock, NoLabelBBoxBlock), 
        get_items=get_image_files,
        get_y=[BBoxTruth(df)],
        n_inp=1,
        item_tfms=[Resize(224, method=ResizeMethod.Pad, pad_mode=PadMode.Border)])

    dls=block.dataloaders(data_url, batch_size=64)
    dls.show_batch(max_n=9, figsize=(8, 8))
    plt.savefig("batch")
