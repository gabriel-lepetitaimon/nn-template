from torch import nn

from .. import Cfg
from ..misc.clip_pad import cat_crop
from .unet import SimpleUnetCfg, SimpleUnet
from ..task import Segmentation2DCfg
from ..task.losses import Loss


class WNetCfg(SimpleUnetCfg):
    auxilary_loss = Cfg.any(Cfg.float(), Cfg.bool(), default=1)

    def create(self, in_channels: int):
        return WNet(self, in_channels)

    def create_unet(self, in_channels: int):
        return super().create_unet(in_channels)


class WNet(nn.Module):
    def __init__(self, cfg: WNetCfg, in_channels: int):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        n_in = in_channels

        self.unet1: SimpleUnet = self.cfg.create_unet(n_in)
        self.unet2: SimpleUnet = self.cfg.create_unet(n_in+self.unet1.n_classes)

    def forward(self, x):
        y1 = self.unet1(x)
        y2 = self.unet2(cat_crop(x, y1))
        return y2

    def create_auxilary_loss(self):
        if not self.cfg.auxilary_loss:
            return None

        alpha_aux = 1 if self.cfg.auxilary_loss is True else self.cfg.auxilary_loss

        seg_task_cfg: Segmentation2DCfg = self.cfg.root()['task']
        if isinstance(seg_task_cfg, Segmentation2DCfg):
            loss_fn: Loss = seg_task_cfg.loss.create()
        else:
            raise NotImplementedError('Auxiliary loss of WNet is only implemented for Segmentation2D tasks')

        unet1_output = None

        def forward_hook(module, input, output):
            nonlocal unet1_output
            if self.training:
                unet1_output = output
            else:
                unet1_output = None

        self.unet1.register_forward_hook(forward_hook)

        def compute_loss(_, target, mask):
            if unet1_output is None:
                return 0
            else:
                return loss_fn(unet1_output, target, mask) * alpha_aux

        return compute_loss
