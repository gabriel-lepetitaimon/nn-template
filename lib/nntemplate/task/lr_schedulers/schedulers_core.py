__all__ = ['SchedulerCfg', 'scheduler']

import torch
from ... import Cfg

_registered_schedulers = {}


class SchedulerCfg(Cfg.Obj):
    type: str = ''

    @classmethod
    def register(cls, name):
        def register(scheduler: SchedulerCfg):
            _registered_schedulers[name] = scheduler
            scheduler.type = name
            return scheduler

        return register

    def create(self, optimizer: torch.optim.Optimizer, last_epoch=-1):
        pass


def scheduler(default=None):
    return Cfg.obj(shortcut='type', obj_types=_registered_schedulers, default=default)