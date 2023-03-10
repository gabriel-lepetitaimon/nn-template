from .. import Cfg
import torch.optim as torch_opti
import inspect


class OptimizerCfg(Cfg.Obj):
    type = Cfg.strMap({
        'Adam': torch_opti.Adam,
        'AdamW': torch_opti.AdamW,
        'Adamax': torch_opti.Adamax,
        'ASGD': torch_opti.ASGD,
        'SGD': torch_opti.SGD,
    })
    lr = Cfg.float()

    def create(self, parameters):
        Optimizer = self.type
        arch_opts = list(inspect.signature(Optimizer).parameters.values())
        cfg = dict()
        attr = self.attr()
        for k, v in self.items():
            if k == 'type' or k in cfg:
                continue
            elif k in attr:
                cfg[k] = attr[k]
            elif k in arch_opts:
                cfg[k] = v
            else:
                print(f'Warning! Useless parameter "{k}" for optimizer {self["type"]}.')
        return Optimizer(parameters, **cfg)
