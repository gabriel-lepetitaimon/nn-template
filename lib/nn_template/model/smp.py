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


@Cfg.register_obj('model', type='smp')
class SMPModelCfg(Cfg.Obj):
    architecture = Cfg.strMap(ARCHITECTURE)
    encoder_name = Cfg.oneOf(*smp.encoders.get_encoder_names(), default='resnet34')
    encoder_depth: int = Cfg.int(min=3, max=5, default=5)
    encoder_weights = Cfg.oneOf('imagenet', 'noisy-student', 'swsl', 'random', None, default='imagenet')
    decoder_use_batchnorm = Cfg.oneOf(True, False, 'inplace', default=True)

    def _after_populate(self):
        try:
            n_classes = self.root()['task'].n_classes
        except:
            raise Cfg.InvalidAttr('Invalid task for SMP Model',
                                  'SMP models requires n-classes to be defined in task.',
                                  mark=self.root().get_mark('task'))

        if n_classes == 'binary':
            raise Cfg.InvalidAttr('Invalid task.n-classes for SMP Model',
                                  "SMP models don't support formal binary segmentation. Use task.n-classes: 2 instead.",
                                  mark=self.root().get_mark('task.n-classes'))

    def create(self, in_channels: int, opt: Mapping[str, any] = None):
        Arch = self.architecture
        arch_opts = list(inspect.signature(Arch).parameters.values())

        n_classes = self.root()['task'].n_classes
        if n_classes == 'binary':
            raise Cfg.InvalidAttr('Invalid task for SMP Model',
                                  'SMP models requires n-classes to be defined in task.',
                                  mark=self.root().get_mark('task'))

        cfg = dict(in_channels=in_channels, classes=n_classes)

        attr = self.attr()
        for k, v in self.items():
            if k == 'architecture' or k in cfg:
                continue
            elif k in attr:
                cfg[k] = attr[k]
            elif k in arch_opts:
                cfg[k] = v
            elif k not in ('type',):
                print(f'Warning! Useless parameter "{k}" for architecture {self["architecture"]}')
        if opt:
            cfg.update({k: v for k, v in opt.items() if k in arch_opts})
        return Arch(**cfg)
