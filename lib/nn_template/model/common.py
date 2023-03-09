import math
import torch
from torch import nn

from ..config import Cfg


class IdentityModule(nn.Module):
    def forward(self, *args):
        return args


########################################################################################################################
class SwitchableNorm(nn.modules.batchnorm._NormBase):
    def __init__(self, num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 device=None,
                 dtype=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        self.lambdas = nn.Parameter(torch.ones((3,), device=device))

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim():d} input)")

    def compute_mean_var(self, input: torch.Tensor):
        instance_var, instance_mean = torch.var_mean(input, dim=(2, 3), keepdim=True)
        batch_mean = torch.mean(instance_mean, dim=0,  keepdim=True)
        batch_var = torch.mean(instance_var, dim=0, keepdim=True)
        layer_mean = torch.mean(instance_mean, dim=1,  keepdim=True)
        layer_var = torch.mean(instance_var, dim=1, keepdim=True)
        return batch_mean, batch_var, instance_mean, instance_var, layer_mean, layer_var

    def forward(self, input):
        self._check_input_dim(input)

        (batch_mean, batch_var,
         instance_mean, instance_var,
         layer_mean, layer_var) = self.compute_mean_var(input)

        if self.training and self.track_running_stats:
            exp_code_factor = 0.0 if self.momentum is None else self.momentum

            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exp_code_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exp_code_factor = self.momentum
            n = input.shape[0]
            self.running_mean.copy_((exp_code_factor * self.running_mean) +
                                    (1.0-exp_code_factor) * batch_mean.view(self.num_features))
            self.running_var.copy_((exp_code_factor * self.running_var) +
                                   (1.0 - exp_code_factor) * (n / (n - 1) * batch_var.view(self.num_features)))
        elif self.track_running_stats:
            batch_mean = self.running_mean.view((1, self.num_features, 1, 1)).expand_as(input)
            batch_var = self.running_var.view((1, self.num_features, 1, 1)).expand_as(input)

        lambdas = nn.functional.softmax(self.lambdas, 0)
        agg_mean = sum(lambd*mean for lambd, mean in zip(lambdas, (batch_mean, instance_mean, layer_mean)))
        agg_var = sum(lambd*var for lambd, var in zip(lambdas, (batch_var, instance_var, layer_var)))

        normed_input = (input-agg_mean) / (agg_var + self.eps).sqrt()

        if self.affine:
            weight = self.weight.view((1, self.num_features, 1, 1)).expand_as(normed_input)
            bias = self.bias.view((1, self.num_features, 1, 1)).expand_as(normed_input)
            return weight * normed_input + bias
        else:
            return normed_input


class NormCfg(Cfg.Obj):
    norm = Cfg.strMap({'instance': nn.InstanceNorm2d,
                       'batch': nn.BatchNorm2d,
                       'layer': nn.LayerNorm,
                       'switchable': SwitchableNorm,
                       })

    eps = Cfg.float(None)
    momentum = Cfg.float(None)
    affine = Cfg.bool(None)
    track_running_stats = Cfg.bool(None)

    def create(self, n_features) -> nn.Module:
        return self.norm(n_features, **{k: v for k, v in self.attr().items() if k != 'norm' and v is not None})


class DownSamplingCfg(Cfg.Obj):
    type = Cfg.oneOf('conv', 'max', 'avg')

    def create(self, n_features, k=2) -> nn.Module:
        match self.type:
            case 'conv':
                return nn.Conv2d(n_features, n_features, kernel_size=k, stride=k,
                                 padding=math.ceil(k / 2), padding_mode='reflect')
            case 'max':
                return nn.MaxPool2d(k)
            case 'avg':
                return nn.AvgPool2d(k)


class UpSamplingCfg(Cfg.Obj):
    type = Cfg.oneOf('conv', 'bilinear', 'nearest', default='bilinear')

    def create(self, n_features, k=2) -> nn.Module:
        match self.type:
            case 'conv':
                return nn.ConvTranspose2d(n_features, n_features,
                                          kernel_size=k, stride=k,
                                          padding=math.ceil(k / 2), padding_mode='reflect')
            case 'bilinear':
                return nn.UpsamplingBilinear2d(scale_factor=k)
            case 'nearest':
                return nn.UpsamplingNearest2d(scale_factor=k)
