import torch
from torch import nn

from .task import Cfg, LightningTask, LightningTaskCfg, Loss, loss_attr, OptimizerCfg
from .metrics import metrics_attr
from ..nn_misc.clip_pad import clip_pad_center, select_pixels_by_mask


@Cfg.register_obj("task", type='Segmentation2D')
class Segmentation2DCfg(LightningTaskCfg):
    n_classes = Cfg.str('binary')
    objective = Cfg.str('acc')
    direction = Cfg.oneOf('max', 'min', default='max')

    loss: Loss = loss_attr(default='cross-entropy')
    optimizer: OptimizerCfg = Cfg.obj(default='Adam', shortcut='type')

    @n_classes.checker
    def check_n_classes(self, value):
        if value == 'binary':
            return value
        return int(value)

    @objective.checker
    def check_objective(self, value):
        value = value.strip()
        if value not in self.metrics + ['val-'+m for m in self.metrics]:
            raise Cfg.InvalidAttr(f'Unknown metric "{value}" provided for attribute {self.name}',
                                  f"Valid metrics are {', '.join(self.metrics)}.")
        return value


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

    def compute_pred_target_loss(self, batch):
        x, target = batch['x'], batch['y']
        mask = batch.get('mask', None)

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

    def _eval_metrics(self, batch, dataset_idx=None):
        pred, target, loss = self.compute_pred_target_loss(batch)
        dataset_name = 'val' if dataset_idx is None else self.test_datasets_names[dataset_idx]

        self.log(dataset_name+'-loss', loss)
        self.compute_metrics(pred, target, dataset=dataset_name, log=True)

    def training_step(self, batch, batch_idx):
        pred, target, loss = self.compute_pred_target_loss(batch)
        self.log('train-loss', loss,
                 on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch):
        self.evaluate_metrics(batch)
        return batch['pred']

    def test_step(self, batch):
        self.evaluate_metrics(batch)
        return batch['pred']
