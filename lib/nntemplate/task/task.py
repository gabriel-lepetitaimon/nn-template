__all__ = ['LightningTask', 'LightningTaskCfg']

import abc
from typing import Tuple
import pytorch_lightning as pl
from torch import nn

from .metrics import metrics_attr
from .. import Cfg
from ..datasets import DatasetsCfg
from .losses import LossCfg, loss_attr
from .optimizer import OptimizerCfg
from .lr_schedulers.schedulers_core import SchedulerCfg, scheduler



class LightningTask(pl.LightningModule):

    @property
    def test_dataloaders_names(self) -> tuple[str]:
        return self.cfg.test_datasets_names


class LightningTaskCfg(Cfg.Obj):

    metrics = metrics_attr('acc', 'classification')

    @property
    def test_datasets_names(self) -> Tuple[str] | Tuple[()]:
        datasets_cfg: DatasetsCfg = self.root()['datasets']
        if not isinstance(datasets_cfg, DatasetsCfg):
            return ()
        return datasets_cfg.test_datasets_names

    @property
    def metrics_names(self):
        metrics = [m.strip() for m in self.root().get('task.metrics').split(',')]+['loss']
        metrics = [f'{p}-{m}' for m in metrics for p in ('val',)+self.test_datasets_names]
        metrics += ['train-loss']
        return metrics

    @abc.abstractmethod
    def create_task(self, model: nn.Module | None = None) -> LightningTask: ...