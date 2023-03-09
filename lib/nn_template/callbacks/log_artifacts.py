import math
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
import wandb

from ..misc.lut import prepare_lut
from ..misc.clip_pad import clip_pad_center


class Export2DLabel(Callback):
    def __init__(self, color_map, dataset_names: list[str] | None = None, every_n_epoch=False, on_test=True):
        super(Export2DLabel, self).__init__()

        self.color_lut = prepare_lut(color_map, bgr=False, source_dtype=int)
        self.dataset_names = dataset_names
        self.every_n_epoch = every_n_epoch
        self.on_test = on_test
        self._last_epoch = -100

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if self.every_n_epoch and pl_module.current_epoch-self._last_epoch >= self.every_n_epoch:
            self._last_epoch = pl_module.current_epoch

            try:
                dataloader_name = self.dataset_names[dataloader_idx]
            except:
                dataloader_name = f'val-{dataloader_idx}' if dataloader_idx is not None else 'val'
            self.export_batch(batch, batch_idx, dataloader_name)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.on_test:
            try:
                dataloader_name = self.dataset_names[dataloader_idx]
            except:
                dataloader_name = f'test-{dataloader_idx}' if dataloader_idx is not None else 'test'
            self.export_batch(batch, batch_idx, dataloader_name)

    def format_batch(self, batch):
        y = (batch['y'] != 0).float()
        y_pred = batch['y_pred']
        if y_pred.dtype == torch.float:
            y_pred = y_pred > 0.5

        y = clip_pad_center(y, y_pred.shape)

        if 'mask' in batch:
            mask = clip_pad_center(batch['mask'], y_pred.shape)
            y[mask == 0] = float('nan')

        diff = torch.stack((y, y_pred), dim=1)
        diff = diff.cpu().numpy()

        return [self.color_lut(img).transpose(1, 2, 0) for img in diff]

    def export_batch(self, batch, batch_idx, dataloader_name):
        if batch_idx:
            return

        imgs = self.format_batch(batch)
        img = make_grid([img for img in imgs], nrow=2)
        wandb.log({dataloader_name: wandb.Image(img,)})


def make_grid(imgs: list[np.ndarray], nrow=1, padding=5, resize='max') -> np.ndarray:
    h, w = max(img.shape[0] for img in imgs), max(img.shape[1] for img in imgs)
    c = imgs[0].shape[2]
    nrow = min(nrow, len(imgs))
    ncol = math.ceil(len(imgs)/nrow)

    h_pad, w_pad = h+padding, w+padding

    grid = np.ones_like(imgs[0], shape=(h_pad*nrow+padding, w_pad*ncol+padding, c))
    for i, img in enumerate(imgs):
        row = i // ncol
        col = i % ncol
        y0, x0 = h_pad*row+padding, w_pad*col + padding
        h0, w0 = img.shape[:2]
        grid[y0:y0+h0, x0:x0+w0] = img
    return grid
