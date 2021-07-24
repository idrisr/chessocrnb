__all__ = ['NoLabelBBoxLabeler', 'BBoxTruth', 'iou', 'NoLabelBBoxBlock']

from fastai.data.all import *
from fastai.vision.all import *
from fastai.interpret import *
from fastcore.transform import *
from datetime import datetime
from pathlib import Path

import chessocr
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('pdf')

class NoLabelBBoxLabeler(Transform):
    """ Bounding box labeler with no label """
    def decode(self, x, **kwargs):
        self.bbox,self.lbls = None,None
        return self._call('decodes', x, **kwargs)

    def decodes(self, x:TensorBBox):
        self.bbox = x
        return self.bbox if self.lbls is None else LabeledBBox(self.bbox, self.lbls)


class BBoxTruth:
    """ get bounding box location from DataFrame """
    def __init__(self, df): self.df=df
    def __call__(self, o):
        size,x,y,*_ =self.df.iloc[int(o.stem)-1]
        return [[x,y, x+size, y+size]]


def iou(pred, target):
    """ Vectorized Intersection Over Union calculation """
    target = Tensor.cpu(target).squeeze(1)
    pred = Tensor.cpu(pred)
    ab = np.stack([pred, target])
    intersect_area = np.maximum(ab[:, :, [2, 3]].min(axis=0) - ab[:, :, [0, 1]].max(axis=0), 0).prod(axis=1)
    union_area = ((ab[:, :, 2] - ab[:, :, 0]) * (ab[:, :, 3] - ab[:, :, 1])).sum(axis=0) - intersect_area
    return (intersect_area / union_area).mean()


NoLabelBBoxBlock = TransformBlock(type_tfms=TensorBBox.create,
                             item_tfms=[PointScaler, NoLabelBBoxLabeler])


def get_learner(dls):
    learn = cnn_learner(dls, resnet18, 
                        metrics=[iou], 
                        loss_func=MSELossFlat())
    learn.model.cuda()
    return learn

def get_dataloader():
    data_url = Path.home()/".fastai/data/chess"
    df = pd.read_csv(data_url/'annotations.csv', index_col=0)
    block = DataBlock(
        blocks=(ImageBlock, NoLabelBBoxBlock), 
        get_items=get_image_files,
        get_y=[BBoxTruth(df)],
        n_inp=1,
        item_tfms=[Resize(224, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros)])

    dls=block.dataloaders(data_url, batch_size=64)
    #  dls.show_batch(max_n=9, figsize=(8, 8))
    return block, dls


@typedispatch
def plot_top_losses(x: TensorImage, y: TensorBBox, samples, outs, raws, losses, nrows=None, ncols=None, figsize=None, **kwargs):
    axs = get_grid(len(samples), nrows=nrows, ncols=ncols, add_vert=1,
            figsize=figsize, title='IOU')
    for ax,s,o,r,l in zip(axs, samples, outs, raws, losses):
        s[0].show(ctx=ax, **kwargs)
        o[0].show(ctx=ax, **kwargs)
        metric = iou(s[1], o[0])
        ax.set_title(f'{metric:.2f}')

@patch
def __repr__(self:DataLoaders):
    return f"after_item:\n\t{self.after_item}" +\
           f"\nbefore_batch:\n\t {self.before_batch}" +\
           f"\nafter_batch:\n\t {self.after_batch}\n"


def interp(learn):
    interp = Interpretation.from_learner(learn)
    return interp


class Model:
    def __init__(self, dls):
        self.learner = cnn_learner(dls, resnet18, 
                            metrics=[iou], 
                            loss_func=L1LossFlat())

    def __enter__(self): return self.load()
    def __exit__(self, exc_type, exc_value, traceback): self.save()

    def load(self, name=None):
        models = get_files(Path("models/"), ".pth")
        if not models: return self.learner
        model = sorted(models, reverse=True)[0]
        return self.learner.load(model.stem)

    def save(self):
        name = datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")
        self.learner.save(name)
        i = interp(self.learner)
        i.plot_top_losses(50)
        plt.savefig(f'{name}-interp')
        del self.learner

if __name__ == '__main__':
    block, dls = get_dataloader()
    with Model(dls) as learner:
        learner.fine_tune(200)
