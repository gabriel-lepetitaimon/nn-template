import traceback

import torch
from torch import nn

from copy import copy
from .task import Cfg, LightningTask, LightningTaskCfg, LossCfg, loss_attr, OptimizerCfg
from .test_time_augment import TestTimeAugmentCfg
from ..misc.clip_pad import clip_pad_center, select_pixels_by_mask
from ..datasets import DatasetsCfg
from .metrics import Metric


@Cfg.register_obj("task", type='Segmentation2D')
class Segmentation2DCfg(LightningTaskCfg):
    classes = Cfg.strList(None)
    n_classes = Cfg.oneOf(Cfg.int(min=1), 'binary', 'auto', default='auto')
    direction = Cfg.oneOf('max', 'min', default='max')

    loss: LossCfg = loss_attr(default='cross-entropy')
    optimizer: OptimizerCfg = Cfg.obj(default='Adam', shortcut='type')
    test_time_augment: TestTimeAugmentCfg = Cfg.obj(default=None, shortcut='alias')

    @LightningTaskCfg.metrics.post_checker
    def check_metrics(self, value):
        for metric in value.values():
            metric: Metric
            if self['n-classes'] != 'binary':
                match metric.metric_name:
                    case 'auroc':
                        raise Cfg.InvalidAttr(f'Invalid metric "auroc" for "{self.fullname}"',
                                              'Area under the ROC cruve may only be computed on binary segmentation task.')

        return value

    @n_classes.post_checker
    def check_n_classes(self, value):
        n_classes = len(self.classes) if self.classes is not None else None
        if value == 'auto':
            if n_classes is None:
                raise Cfg.InvalidAttr('Either n-classes or classes should be defined in the task configuration')
            if n_classes == 1:
                self['classes'] = ['background'] + self.classes
                return 'binary'
            else:
                return n_classes
        elif value == 'binary':
            if n_classes == 1:
                self['classes'] = ['background'] + self.classes
            elif n_classes != 2:
                raise Cfg.InvalidAttr(f"Invalid number of classes names (classes={self.classes}) for binary classification",
                                      f'Either change "classes" to a list of one or two names, or set n-classes: {n_classes}')
        elif value == n_classes+1:
            self['classes'] = ['background'] + self.classes
        elif value != n_classes:
            raise Cfg.InvalidAttr(f"Invalid number of classes names (classes={self.classes}) for {value}-fold classification",
                                  f'Either change "classes" to a list of {value} names, or set n-classes: {n_classes}')
        return value

    def create_net(self, model: nn.Module):
        return Segmentation2D(self, model=model)


# ==================================================================================
class Segmentation2D(LightningTask):
    def __init__(self, cfg: Segmentation2DCfg, model: nn.Module):
        super(Segmentation2D, self).__init__()
        self.cfg = cfg
        self.model = model
        self.loss = self.cfg.loss.create()
        self.metrics_cfg = {f"{dataset}-{metric_name}": metric
                            for dataset in ('val',) + self.test_dataloaders_names
                            for metric_name, metric in self.cfg.metrics.items()}
        self.metrics = {name: metric.create(num_classes=self.cfg.n_classes)
                        for name, metric in self.metrics_cfg.items()}
        for name, metric in self.metrics.items():
            self.add_module(name, metric)

    def forward(self, **batch):

        if self.cfg.test_time_augment:
            merger = self.cfg.test_time_augment.create_merger()
            for transform in self.cfg.test_time_augment.transforms:
                x_aug = transform.augment_image(batch['x'])
                proba = self.model(x_aug, **{k: v for k, v in batch.items() if k not in ('x', 'y', 'mask')})
                if proba.ndim == 3:
                    proba = transform.deaugment_mask(proba.unsqueeze(1)).squeeze(1)
                elif proba.ndim == 4:
                    proba = transform.deaugment_mask(proba)
                merger.append(proba)
            proba = merger.result
        else:
            proba = self.model(batch['x'], **{k: v for k, v in batch.items() if k not in ('x', 'y', 'mask')})

        return self.apply_logits(proba)

    def configure_optimizers(self):
        return self.cfg.optimizer.create(self.parameters())

    def compute_pred_target_loss(self, batch, test_time_augment=False):
        x, target = batch['x'], batch['y']
        mask = batch.get('mask', None)
        binary = self.cfg.n_classes == 'binary'
        logits_in_loss = self.cfg.loss.name == 'cross-entropy' and self.cfg.loss.with_logits

        if test_time_augment and self.cfg.test_time_augment:
            merger = self.cfg.test_time_augment.create_merger()
            for transform in self.cfg.test_time_augment.transforms:
                x_aug = transform.augment_image(x)
                proba = self.model(x_aug, **{k: v for k, v in batch.items() if k not in ('x', 'y', 'mask')})
                if proba.ndim == 3:
                    proba = transform.deaugment_mask(proba.unsqueeze(1)).squeeze(1)
                elif proba.ndim == 4:
                    proba = transform.deaugment_mask(proba)
                merger.append(proba)
            proba = merger.result
        else:
            proba = self.model(x, **{k: v for k, v in batch.items() if k not in ('x', 'y', 'mask')})

        if binary and proba.ndim == 4:
            proba = proba.squeeze(1)
        target = clip_pad_center(target, proba.shape)

        if not logits_in_loss:
            proba = self.apply_logits(proba)

        loss = self.loss(proba, target, mask)

        proba = proba.detach()
        if logits_in_loss:
            proba = self.apply_logits(proba)
        batch['y_pred'] = proba > .5 if binary else torch.argmax(proba, dim=1)

        return loss

    def apply_logits(self, proba: torch.Tensor) -> torch.Tensor:
        if self.cfg.n_classes != 'binary':
            return torch.nn.functional.softmax(proba, dim=1)
        else:
            return torch.nn.functional.sigmoid(proba)

    def compute_metrics(self, batch, dataset='val', log=False):
        for name, metric_cfg in self.metrics_cfg.items():
            if dataset is not None and name.startswith(dataset):
                metric = self.metrics[name]
                args = metric_cfg.prepare_data(batch['y_pred'], batch['y'], batch.get('mask', None))
                try:
                    metric(*args)
                except Exception:
                    e = f'!!! ERROR WHEN COMPUTING METRIC {name} !!!'
                    print('\n\n' + '!' * len(e))
                    print(e)
                    print('\t'+traceback.format_exc().replace('\n', '\n\t'))
                    print('!' * len(e) + '\n\n')
                else:
                    if log:
                        metric_cfg.log(self, name, metric)

    def evaluate_model(self, batch, dataset_name='', test_time_augment=False):
        loss = self.compute_pred_target_loss(batch, test_time_augment=test_time_augment)
        if dataset_name:
            dataset_name += '-'

        self.log(dataset_name+'loss', loss)
        self.compute_metrics(batch, dataset=dataset_name, log=True)

    def training_step(self, batch, batch_idx):
        loss = self.compute_pred_target_loss(batch)
        self.log('train-loss', loss, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate_model(batch, dataset_name='val', test_time_augment=True)
        return batch

    def test_step(self, batch, batch_idx, dataloader_id=None):
        dataloader_name = self.test_dataloaders_names
        if not isinstance(dataloader_name, tuple) or len(dataloader_name) == 0:
            dataloader_name = 'test'
        elif dataloader_id is None:
            dataloader_name = dataloader_name[0]
        elif len(dataloader_name) > dataloader_id:
            dataloader_name = dataloader_name[dataloader_id]
        else:
            raise RuntimeError
        self.evaluate_model(batch, dataset_name=dataloader_name, test_time_augment=True)
        return batch

    @property
    def test_dataloaders_names(self) -> tuple[str]:
        datasets_cfg: DatasetsCfg = self.cfg.root()['datasets']
        return tuple(datasets_cfg.test.keys())
