import torch
from torch import nn

from copy import copy
from .task import Cfg, LightningTask, LightningTaskCfg, Loss, loss_attr, OptimizerCfg
from .test_time_augment import TestTimeAugmentCfg
from ..misc.clip_pad import clip_pad_center, select_pixels_by_mask


@Cfg.register_obj("task", type='Segmentation2D')
class Segmentation2DCfg(LightningTaskCfg):
    n_classes = Cfg.str('binary')
    direction = Cfg.oneOf('max', 'min', default='max')

    loss: Loss = loss_attr(default='cross-entropy')
    optimizer: OptimizerCfg = Cfg.obj(default='Adam', shortcut='type')
    test_time_augment: TestTimeAugmentCfg = Cfg.obj(default=None, shortcut='alias')

    @n_classes.checker
    def check_n_classes(self, value):
        if value == 'binary':
            return value
        return int(value)


# ==================================================================================
class Segmentation2D(LightningTask):
    def __init__(self, cfg: Segmentation2DCfg, model: nn.Module, test_datasets_names=('test',)):
        super(Segmentation2D, self).__init__()
        self.cfg = cfg
        self.model = model
        self.test_datasets_names = test_datasets_names
        self.loss = self.cfg.loss.create(n_classes=self.cfg.n_classes)
        self.metrics_cfg = {f"{dataset}-{metric_name}": metric
                            for dataset in ('val',) + test_datasets_names
                            for metric_name, metric in self.cfg.metrics.items()}
        self.metrics = {name: metric.create(num_classes=self.cfg.n_classes)
                        for name, metric in self.metrics_cfg.items()}

    def forward(self, *args, **kwargs):
        if self.cfg.n_classes == 'binary':
            return torch.sigmoid(self.model(*args, **kwargs))

    def compute_pred_target_loss(self, batch, test_time_augment=False):
        x, target = batch['x'], batch['y']
        mask = batch.get('mask', None)

        if test_time_augment:
            merger = self.cfg.test_time_augment.create_merger()
            for transform in self.cfg.test_time_augment.transforms:
                x = transform.augment_image(x)
                pred = self.model(x, **{k: v for k, v in batch.items() if k not in ('x', 'y', 'mask')})
                if pred.ndim == 3:
                    pred = transform.deaugment_mask(pred)
                elif pred.ndim == 4:
                    pred = transform.deaugment_label(pred)
                merger.append(pred)
            pred = merger.result
        else:
            pred = self.model(x, **{k: v for k, v in batch.items() if k not in ('x', 'y', 'mask')})

        batch['pred'] = pred
        if self.cfg.n_classes == 'binary' and pred.ndim == 4:
            pred = pred.squeeze(1)
        target = clip_pad_center(target, pred.shape)

        pred, target = select_pixels_by_mask(pred, target, mask=mask)
        loss = self.loss(pred, target)

        return pred, target, loss

    def compute_metrics(self, pred, target, dataset='val', log=False):
        for name, metric in self.metrics:
            if name.startswith(dataset):
                metric(pred, target)
                if log:
                    self.metrics_cfg[name].log(name, metric)

    def evaluate_model(self, batch, dataset_name='', test_time_augment=False):
        pred, target, loss = self.compute_pred_target_loss(batch, test_time_augment=test_time_augment)
        if dataset_name:
            dataset_name += '-'

        self.log(dataset_name+'loss', loss)
        self.compute_metrics(pred, target, dataset=dataset_name, log=True)

    def training_step(self, batch, batch_idx):
        pred, target, loss = self.compute_pred_target_loss(batch)
        self.log('train-loss', loss,
                 on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch):
        self.evaluate_metrics(batch, dataset_name='val', test_time_augment=True)
        return batch    # ['pred']

    def test_step(self, batch, dataloader_id):
        dataloader_name = getattr(self.trainer, 'test_dataloaders_name', 'test')
        if isinstance(dataloader_name, list) and len(dataloader_name) > dataloader_id:
            dataloader_name = dataloader_name[dataloader_id]
        self.evaluate_model(batch, dataset_name=dataloader_name, test_time_augment=True)
        return batch    # ['pred']
