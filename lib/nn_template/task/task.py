from functools import cached_property
import pytorch_lightning as pl

from .metrics import metrics_attr
from ..config import Cfg
from .losses import Loss, loss_attr
from .optimizer import OptimizerCfg


class LightningTaskCfg(Cfg.Obj):

    metrics = metrics_attr('acc', 'classification')

    def _test_dataset_names(self) -> tuple[str]:
        return ('test',) # tuple(self.root()['datasets'].test.keys())

    @cached_property
    def metrics_names(self):
        metrics = [m.strip() for m in self.root().get('task.metrics').split(',')]
        metrics = [f'{p}-{m}' for m in metrics for p in ('val',)+self._test_dataset_names()]
        metrics += ['train-loss']
        return metrics


class LightningTask(pl.LightningModule):
    pass