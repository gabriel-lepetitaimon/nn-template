__all__ = ['ExportValidation']

import numpy as np
from pytorch_lightning.callbacks import Callback


class ExportValidation(Callback):
    def __init__(self, color_map, path, dataset_names):
        super(ExportValidation, self).__init__()

        from .lut import prepare_lut
        self.color_lut = prepare_lut(color_map, source_dtype=np.int)
        self.dataset_names = dataset_names
        self.path = path

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.export_batch(batch, outputs, batch_idx, dataloader_idx, prefix='val')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.export_batch(batch, outputs, batch_idx, dataloader_idx, prefix='test')

    def export_batch(self, batch, outputs, batch_idx, dataloader_idx, prefix):
        import os
        import cv2
        import torch
        from steered_cnn.utils import clip_pad_center

        if batch_idx:
            return

        x = batch['x']
        y = (batch['y'] != 0).float()
        y_pred = outputs.detach() > .5
        y = clip_pad_center(y, y_pred.shape)

        if 'mask' in batch:
            mask = clip_pad_center(batch['mask'], y_pred.shape)
            y[mask==0] = float('nan')

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        diff = torch.stack((y, y_pred), dim=1)
        diff = diff.cpu().numpy()
        for i, diff_img in enumerate(diff):
            diff_img = (self.color_lut(diff_img).transpose(1, 2, 0) * 255).astype(np.uint8)
            path = os.path.abspath(os.path.join(self.path, f'{prefix}-{self.dataset_names[dataloader_idx]}-{i}.png'))
            cv2.imwrite(path, diff_img)
