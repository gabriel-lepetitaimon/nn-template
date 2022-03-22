import inspect
import segmentation_models_pytorch as smp
from typing import Mapping

from ..config import Cfg


ARCHITECTURE = {
    'Unet': smp.Unet,
    'Unet++': smp.UnetPlusPlus,
    'UnetPlusPlus': smp.UnetPlusPlus,
    'MAnet': smp.MAnet,
    'Linknet': smp.Linknet,
    'FPN': smp.FPN,
    'PSPNet': smp.PSPNet,
    'PAN': smp.PAN,
    'DeepLabV3': smp.DeepLabV3,
    'DeepLabV3+': smp.DeepLabV3Plus,
    'DeepLabV3Plus': smp.DeepLabV3Plus,
}


@Cfg.register_obj('model')
class Model(Cfg.Obj):
    architecture = Cfg.strMap(ARCHITECTURE)
    encoder_name: str = 'resnet34'
    encoder_depth: int = Cfg.int(min=3, max=5, default=5)
    encoder_weights: str = 'imagenet'
    decoder_use_batchnorm = Cfg.oneOf(True, False, 'inplace', default=True)

    def model(self, in_channels: int, classes: int, activation: str = None, opt: Mapping[str, any] = None):
        Arch = self.architecture
        arch_opts = list(inspect.signature(Arch).parameters.values())
        cfg = dict(in_channels=in_channels, classes=classes, activation=activation)
        attr = self.attr()
        for k, v in self.items():
            if k == 'architecture' or k in cfg:
                continue
            elif k in attr:
                cfg[k] = attr[k]
            elif k in arch_opts:
                cfg[k] = v
            else:
                print(f'Warning! Useless parameter "{k}" for architecture {self["architecture"]}')
        if opt:
            cfg.update({k: v for k, v in opt.items() if k in arch_opts})
        return Arch(**cfg)
