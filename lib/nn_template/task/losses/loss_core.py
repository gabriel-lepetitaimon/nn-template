import torch
from inspect import signature

from ...config import Cfg
from ...config.cfg_dict import UNDEFINED


class Loss:
    def __init__(self, cfg, loss_fn):
        self.loss_fn = loss_fn
        self.cfg = cfg

    def __call__(self, pred, target, mask=None):
        return self.loss_fn(pred=pred, target=target, mask=mask)


class LossCfg(Cfg.Obj):
    name = None

    def create(self) -> Loss:
        loss_fn = self.loss
        loss_args = signature(loss_fn).parameters
        if not loss_args:
            loss_fn = loss_fn()
        return Loss(self, loss_fn) if not isinstance(loss_fn, Loss) else loss_fn

    def loss(self, **kwargs):
        raise NotImplementedError


def loss_attr(default=UNDEFINED, nullable=None):
    return Cfg.obj(default=default, obj_types=_LOSSES, nullable=nullable)


_LOSSES = {}


def register_loss(name):
    global _LOSSES

    def register(f_loss: LossCfg):
        _LOSSES[name] = f_loss
        f_loss.name = name
        return f_loss

    return register

