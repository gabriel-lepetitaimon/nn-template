import traceback

import torch
from torch import nn

from copy import copy
from .task import Cfg, LightningTask, LightningTaskCfg, Loss, loss_attr, OptimizerCfg
from .test_time_augment import TestTimeAugmentCfg
from ..misc.clip_pad import clip_pad_center, select_pixels_by_mask
from ..datasets import DatasetsCfg
from .metrics import Metric


@Cfg.register_obj("task", type='Segmentation2D')
class Segmentation2DCfg(LightningTaskCfg):
    n_classes = Cfg.oneOf(Cfg.int(min=1), 'binary')
    direction = Cfg.oneOf('max', 'min', default='max')

    loss: Loss = loss_attr(default='cross-entropy')
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

    @n_classes.checker
    def check_n_classes(self, value):
        if value == 'binary':
            return value
        return int(value)

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

    def forward(self, *args, **kwargs):
        pred = self.model(*args, **kwargs)
        if self.cfg.n_classes == 'binary':
            return torch.sigmoid(pred)
        else:
            return torch.argmax(pred, dim=1)

    def configure_optimizers(self):
        return self.cfg.optimizer.create(self.parameters())

    def compute_pred_target_loss(self, batch, test_time_augment=False):
        x, target = batch['x'], batch['y']
        mask = batch.get('mask', None)

        if test_time_augment and self.cfg.test_time_augment:
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

        batch['pred'] = pred.detach()
        if self.cfg.n_classes == 'binary' and pred.ndim == 4:
            pred = pred.squeeze(1)
        target = clip_pad_center(target, pred.shape)

        if self.cfg.n_classes == 'binary':
            pred, target = select_pixels_by_mask(pred, target, mask=mask)
        else:
            pred = torch.nn.functional.softmax(pred, dim=1)

            target[~mask] = -1
            batch['pred'] = torch.argmax(batch['pred'], dim=1)
        loss = self.loss(pred, target)

        return pred, target, loss

    def compute_metrics(self, pred, target, dataset='val', log=False):
        for name, metric_cfg in self.metrics_cfg.items():
            if name.startswith(dataset):
                metric = self.metrics[name]
                pred, target = metric_cfg.prepare_data(pred, target)
                try:
                    metric(pred, target)
                except Exception as e:
                    e = f'!!! ERROR WHEN COMPUTING METRIC {name} !!!'
                    print('\n\n' + '!' * len(e))
                    print(e)
                    print('\t'+traceback.format_exc().replace('\n', '\n\t'))
                    print('!' * len(e) + '\n\n')
                else:
                    if log:
                        metric_cfg.log(self, name, metric)

    def evaluate_model(self, batch, dataset_name='', test_time_augment=False):
        pred, target, loss = self.compute_pred_target_loss(batch, test_time_augment=test_time_augment)
        if dataset_name:
            dataset_name += '-'

        self.log(dataset_name+'loss', loss)
        if self.cfg.n_classes != 'binary':
            pred = torch.argmax(pred, dim=1)
            target = target.to(torch.int)
        self.compute_metrics(pred, target, dataset=dataset_name, log=True)

    def training_step(self, batch, batch_idx):
        pred, target, loss = self.compute_pred_target_loss(batch)
        self.log('train-loss', loss,
                 on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate_model(batch, dataset_name='val', test_time_augment=True)
        return batch    # ['pred']

    def test_step(self, batch, batch_idx, dataloader_id=None):
        dataloader_name = self.test_dataloaders_names
        if not isinstance(dataloader_name, list) or len(dataloader_name) == 0:
            dataloader_name = 'test'
        elif dataloader_id is None:
            dataloader_name = dataloader_name[0]
        elif len(dataloader_name) > dataloader_id:
            dataloader_name = dataloader_name[dataloader_id]
        else:
            raise RuntimeError
        self.evaluate_model(batch, dataset_name=dataloader_name, test_time_augment=True)
        return batch    # ['pred']

    @property
    def test_dataloaders_names(self) -> tuple[str]:
        datasets_cfg: DatasetsCfg = self.cfg.root()['datasets']
        return tuple(datasets_cfg.test.keys())
