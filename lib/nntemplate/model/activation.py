from torch import nn
from ..config import Cfg, UNDEFINED


class ActivationCfg(Cfg.Obj):
    name = None

    def create(self) -> nn.Module:
        from .common import IdentityModule
        return IdentityModule()


def activation_attr(default=UNDEFINED, nullable=None):
    return Cfg.obj(default=default, obj_types=_ACTIVATIONS, nullable=nullable)


_ACTIVATIONS = {}


def register_activation(name):
    global _ACTIVATIONS

    def register(activation: ActivationCfg):
        _ACTIVATIONS[name] = activation
        activation.name = name
        return activation
    return register


########################################################################################################################
@register_activation('relu')
class ReLUCfg(ActivationCfg):
    leaky = Cfg.oneOf(Cfg.bool(), Cfg.float(0,min=0), default=False)

    def create(self) -> nn.Module:
        if self.leaky:
            leaky = self.leaky if isinstance(self.leaky, float) else None
            return nn.LeakyReLU(negative_slope=leaky)
        return nn.ReLU()


@register_activation('tanh')
class TanhCfg(ActivationCfg):
    def create(self) -> nn.Module:
        return nn.Tanh()


@register_activation('swish')
class SwishCfg(ActivationCfg):
    def create(self) -> nn.Module:
        return nn.Hardswish()


@register_activation('hardshrink')
class ShrinkCfg(ActivationCfg):
    beta = Cfg.float(0.5)
    type = Cfg.oneOf('hard', 'soft', default='hard')

    def create(self) -> nn.Module:
        match self.type:
            case 'hard': return nn.Hardshrink(lambd=self.beta)
            case 'soft': return nn.Softshrink(lambd=self.beta)
