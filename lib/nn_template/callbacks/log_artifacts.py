import numpy as np
import torch
import wandb
from pytorch_lightning.callbacks import Callback
from ..misc.lut import prepare_lut
from ..misc.clip_pad import clip_pad_center


class Export2DLabel(Callback):
    def __init__(self, color_map, dataset_names, on_validation=False, on_test=True):
        super(Export2DLabel, self).__init__()

        self.color_lut = prepare_lut(color_map, source_dtype=np.int)
        self.dataset_names = dataset_names
        self.on_validation = on_validation
        self.on_test = on_test

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.on_validation:
            self.export_batch(batch, outputs, batch_idx, dataloader_idx, prefix='val')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.on_test:
            self.export_batch(batch, outputs, batch_idx, dataloader_idx, prefix='test')

    def export_batch(self, batch, outputs, batch_idx, dataloader_name, prefix):
        if batch_idx:
            return

        if dataloader_name:
            dataloader_name += '-'

        x = batch['x']
        y = (batch['y'] != 0).float()
        y_pred = outputs.detach() > .5
        y = clip_pad_center(y, y_pred.shape)

        if 'mask' in batch:
            mask = clip_pad_center(batch['mask'], y_pred.shape)
            y[mask == 0] = float('nan')

        diff = torch.stack((y, y_pred), dim=1)
        diff = diff.cpu().numpy()
        for i, diff_img in enumerate(diff):
            diff_img = self.color_lut(diff_img).transpose(1, 2, 0)
            img = wandb.Image(diff_img, )
            wandb.log({f'{prefix}-{dataloader_name}{i}': img})
