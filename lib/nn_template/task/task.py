__all__ = ['LightningTask', 'LightningTaskCfg']

from typing import Tuple
import pytorch_lightning as pl

from .metrics import metrics_attr
from .. import Cfg
from ..datasets import DatasetsCfg
from .losses import LossCfg, loss_attr
from .optimizer import OptimizerCfg
from .lr_schedulers.schedulers_core import SchedulerCfg, scheduler


class LightningTaskCfg(Cfg.Obj):

    metrics = metrics_attr('acc', 'classification')

    def _test_dataset_names(self) -> Tuple[str] | Tuple[()]:
        datasets_cfg: DatasetsCfg = self.root()['datasets']
        if not isinstance(datasets_cfg, DatasetsCfg):
            return ()
        return tuple(datasets_cfg.test.keys())

    @property
    def metrics_names(self):
        metrics = [m.strip() for m in self.root().get('task.metrics').split(',')]+['loss']
        metrics = [f'{p}-{m}' for m in metrics for p in ('val',)+self._test_dataset_names()]
        metrics += ['train-loss']
        return metrics


class LightningTask(pl.LightningModule):
    pass
