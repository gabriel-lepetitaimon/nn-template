__all__ = ['CosineAnnealingCfg', 'TanhCfg']

import torch
import timm.scheduler as timm_schedulers

from ... import Cfg
from .schedulers_core import SchedulerCfg


@SchedulerCfg.register('cosine-annealing')
class CosineAnnealingCfg(SchedulerCfg):
    t_initial = Cfg.int(10, min=0)
    lr_min = Cfg.float(1e-7)               # Minimum lr
    cycle_mul = Cfg.float(None, min=0)     # Factor increasing the number of epoch in the next cycle.
    cycle_decay = Cfg.float(None, min=0)   # Factor multiplicating the starting lr of the next cycle.
    cycle_limit = Cfg.int(None, min=0)     # Maximum number of cycle
    warmup_t = Cfg.int(None, min=0)             # Number of epochs to warmup
    warmup_lr_init = Cfg.float(None, min=0)     # Initial learning rate during warmup
    warmup_prefix = Cfg.bool(None)              # Remove warmup epochs from the cycle
    t_in_epochs = Cfg.bool(None)
    noise_range_t = Cfg.bool(None)              # Add noise to the lr range
    noise_pct = Cfg.float(None, min=0, max=1)   # Amount of noise to add to the lr range. Default: 0.67
    noise_std = Cfg.float(None, min=0)          # Standard deviation of the noise. Default: 1.0
    k_decay = Cfg.float(None, min=0)

    def create(self, optimizer: torch.optim.Optimizer, last_epoch=-1):
        defaults = dict(cycle_limit=1e12)
        kwargs = defaults | {k: v for k, v in self.attr().items() if k not in ('type',) and v is not None}
        scheduler = timm_schedulers.CosineLRScheduler(optimizer=optimizer,
                                                 **kwargs,
                                                 noise_seed=self.root().get('training.seed', 42),
                                                 initialize=True)
        return {'scheduler': scheduler, 'interval': 'epoch'}


@SchedulerCfg.register('tanh')
class TanhCfg(SchedulerCfg):
    t_initial = Cfg.int(10, min=0)
    lr_min = Cfg.float(0)               # Minimum lr
    lb = Cfg.float(0)                   # The lower bound of the tanh function.
    ub = Cfg.float(1)
    cycle_mul = Cfg.float(1, min=0)     # Factor increasing the number of epoch in the next cycle.
    cycle_decay = Cfg.float(1, min=0)

    warmup_t = 0                        # Number of epochs to warmup
    warmup_lr_init = 0                  # Initial learning rate during warmup
    warmup_prefix = False               # Remove warmup epochs from the total number of epochs
    t_in_epochs = True

    def create(self, optimizer: torch.optim.Optimizer, last_epoch=-1):
        scheduler = timm_schedulers.TanhLRScheduler(optimizer=optimizer,
                                               **{k: v for k, v in self.attr().items() if k not in ('type',)},
                                               noise_seed=self.root().get('training.seed', 42),
                                               initialize=True)
        return {'scheduler': scheduler, 'interval': 'epoch'}
