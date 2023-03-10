__all__ = ['Segmentation2DCfg', 'Segmentation2D']

from copy import copy
import traceback
import torch
import wandb
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from torch import nn


from ..misc.function_tools import match_params
from .task import Cfg, LightningTask, LightningTaskCfg, LossCfg, loss_attr, OptimizerCfg, SchedulerCfg, scheduler
from .test_time_augment import TestTimeAugmentCfg
from ..misc.clip_pad import clip_pad_center, select_pixels_by_mask
from ..datasets import DatasetsCfg
from .metrics import MetricCfg, Metric


@Cfg.register_obj("task", type='Segmentation2D')
class Segmentation2DCfg(LightningTaskCfg):
    classes = Cfg.strList()
    n_classes = Cfg.oneOf(Cfg.int(min=1), 'binary', 'auto', default='auto')
    direction = Cfg.oneOf('max', 'min', default='max')

    loss: LossCfg = loss_attr(default='cross-entropy')
    optimizer: OptimizerCfg = Cfg.obj(default='Adam', shortcut='type')
    scheduler: SchedulerCfg = scheduler()
    test_time_augment: TestTimeAugmentCfg = Cfg.obj(default=None, shortcut='alias')

    @LightningTaskCfg.metrics.post_checker
    def check_metrics(self, value):
        for metric in value.values():
            metric: MetricCfg
            if self['n-classes'] not in ('binary', 2):
                match metric.metric_name:
                    case 'auroc':
                        raise Cfg.InvalidAttr(f'Invalid metric "auroc" for "{self.fullname}"',
                                              'Area under the ROC cruve may only be computed on binary segmentation task.')

        return value

    @classes.post_checker
    def check_classes(self, value):
        n_value = len(value)
        n_classes = self.get('n-classes', 'auto')
        if n_classes == 'auto':
            if n_value == 1:
                self['n-classes'] = 'binary'
                return ['background'] + value
            else:
                self['n-classes'] = n_value
                return n_value
        elif n_classes == 'binary':
            if n_value == 1:
                return ['background'] + value
            elif n_value != 2:
                raise Cfg.InvalidAttr(f"Invalid number of classes names for binary classification",
                                      f'Either change "classes" to a list of one or two names, or set n-classes: {n_value}.')
        elif n_classes == n_value+1:
            return ['background'] + value
        elif n_classes != n_value:
            raise Cfg.InvalidAttr(f"Invalid number of classes names for {n_classes}-fold classification",
                                  f'Either change "classes" to a list of {n_classes} names, or set n-classes: {n_value}')
        return value

    @n_classes.post_checker
    def check_n_classes(self, value):
        try:
            n_classes = len(self.classes)
        except AttributeError:
            return value
        match value:
            case 'binary':
                if n_classes in (1, 2):
                    return value
            case 'auto': return value
            case int():
                if n_classes in (value, value+1):
                    return value
        raise Cfg.InvalidAttr(f"Invalid number of classes names for {value}-fold classification",
                              f'Either change "classes" to a list of {value} names, or set n-classes: {n_classes}')

    def create_net(self, model: nn.Module):
        return Segmentation2D(self, model=model)

    @property
    def logits_in_loss(self):
        return self.loss.name == 'cross-entropy' and self.loss.with_logits

    def apply_logits(self, proba: torch.Tensor) -> torch.Tensor:
        if self.n_classes != 'binary':
            return torch.nn.functional.softmax(proba, dim=1)
        else:
            return torch.nn.functional.sigmoid(proba)


# ==================================================================================
class Segmentation2D(LightningTask):
    def __init__(self, cfg: Segmentation2DCfg, model: nn.Module):
        super(Segmentation2D, self).__init__()
        self.cfg = cfg
        self.datasets_cfg: DatasetsCfg = self.cfg.root()['datasets']
        self.model = model
        self.loss = self.cfg.loss.create()
        self.auxilary_loss = model.create_auxilary_losses() if hasattr(model, 'create_auxilary_losses') else None


        self.metrics_cfg = {f"{dataset}-{metric_name}": metric
                            for dataset in ('val', 'test')
                            for metric_name, metric in self.cfg.metrics.items()}
        self.metrics: dict[str, Metric] = {name: metric.create(num_classes=self.cfg.n_classes)
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
        optim = self.cfg.optimizer.create(self.parameters())
        scheduler = self.cfg.scheduler.create(optim) if self.cfg.scheduler else None
        if scheduler:
            return [optim], [scheduler]
        else:
            return optim

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, optimizer_idx: int, metric) -> None:
        match_params(scheduler.step, metrics=metric, epoch=self.current_epoch, step=self.global_step)
        wandb.log({'lr': scheduler.optimizer.param_groups[0]['lr']})

    def compute_pred_target_loss(self, batch, test_time_augment=False):
        x, target = batch['x'], batch['y']
        mask = batch.get('mask', None)
        binary = self.cfg.n_classes == 'binary'

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

        if not self.cfg.logits_in_loss:
            proba = self.cfg.apply_logits(proba)

        loss = self.loss(proba, target, mask)
        if self.auxilary_loss:
            loss += self.auxilary_loss(proba, target, mask)

        proba = proba.detach()
        if self.cfg.logits_in_loss:
            proba = self.cfg.apply_logits(proba)
        batch['y_pred'] = proba > .5 if binary else torch.argmax(proba, dim=1)

        return loss

    def compute_metrics(self, batch, dataset='val', log=False, test=False):
        metric_prefix = 'test' if test else dataset

        if test:
            ignored_metrics = self.datasets_cfg.test[dataset].ignore_metrics
        elif dataset == 'val':
            ignored_metrics = self.datasets_cfg.validate.ignore_metrics
        else:
            ignored_metrics = []

        for name, metric_cfg in self.metrics_cfg.items():
            metric_cfg: MetricCfg
            if name.startswith(metric_prefix) and metric_cfg.name not in ignored_metrics:
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
                        if test and name.startswith('test'):
                            name = dataset+name[4:]
                        metric_cfg.log(self, name, metric)

    def evaluate_model(self, batch, dataset_name='', test_time_augment=False, test=False):
        loss = self.compute_pred_target_loss(batch, test_time_augment=test_time_augment)

        self.compute_metrics(batch, dataset=dataset_name, log=True, test=test)
        if dataset_name:
            dataset_name += '-'
        self.log(dataset_name+'loss', loss)

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
        self.evaluate_model(batch, dataset_name=dataloader_name, test_time_augment=True, test=True)
        return batch

    def on_test_epoch_start(self) -> None:
        for metric_name, metric in self.metrics.items():
            if metric_name.startswith('test'):
                metric.reset()

    @property
    def test_dataloaders_names(self) -> tuple[str]:
        datasets_cfg: DatasetsCfg = self.cfg.root()['datasets']
        return tuple(datasets_cfg.test.keys())
