__all__ = ['ReduceOnPlateauCfg']

import torch
from torch.optim import lr_scheduler as torch_schedulers


from ... import Cfg
from .schedulers_core import SchedulerCfg
from ..metrics import MonitoredMetricCfg


@SchedulerCfg.register('reduce-on-plateau')
class ReduceOnPlateauCfg(SchedulerCfg):
    monitor: MonitoredMetricCfg = Cfg.obj(None, shortcut='metric', nullable=True)
    factor = Cfg.float(0.1, min=0)
    patience = Cfg.int(5, min=1)
    threshold = Cfg.float(1e-4)
    threshold_mode = Cfg.oneOf('rel', 'abs', default='rel')
    cooldown = Cfg.int(0, min=0)
    min_lr = Cfg.float(1e-8, min=0)
    eps = Cfg.float(1e-8)

    def create(self, optimizer: torch.optim.Optimizer, last_epoch=-1):
        from ...training import TrainingCfg
        training_cfg: TrainingCfg = self.root()['training']
        monitor: MonitoredMetricCfg = self.monitor if self.monitor is not None else training_cfg.objective_ckpt_cfg
        scheduler = torch_schedulers.ReduceLROnPlateau(optimizer=optimizer, mode=monitor.mode,
                                                       **{k: v for k, v in self.attr().items()
                                                          if k not in ('type', 'monitor') and v is not None})
        return {'scheduler': scheduler, 'interval': 'epoch', 'monitor': monitor.metric}
