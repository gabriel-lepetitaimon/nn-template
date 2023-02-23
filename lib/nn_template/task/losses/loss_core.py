from ...config import Cfg
from ...config.cfg_dict import UNDEFINED
import torch.nn.functional as F

# TODO: multiple obj_types for one Cfg.obj()


class Loss(Cfg.Obj):
    def create(self):
        raise NotImplementedError


def loss_attr(default=UNDEFINED, nullable=None):
    return Cfg.obj(default=default, obj_types=_LOSSES, nullable=nullable)


_LOSSES = {}


def register_loss(name):
    global _LOSSES

    def register(f_loss):
        _LOSSES[name] = f_loss

    return register

