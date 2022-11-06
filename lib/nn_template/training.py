from . import Cfg


@Cfg.register_obj("training")
class TrainingCfg(Cfg.Obj):
    seed = 1234
    minibatch = Cfg.int()
    max_epoch = Cfg.int()

    def configure_seed(self):
        import random
        import numpy as np
        import torch
        import os

        if self.seed == 0:
            self.seed = int.from_bytes(os.getrandom(32), 'little', signed=False)

        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


@Cfg.register_obj("task")
class TaskCfg(Cfg.Obj):
    type = Cfg.oneOf('img-seg', 'img-clas')
    n_classes = Cfg.str('binary')
    metrics = Cfg.strList('acc')
    optimize = Cfg.str('acc')
    direction = Cfg.oneOf('max', 'min', default='max')

    @n_classes.checker
    def check_n_classes(self, value):
        if value == 'binary':
            return value
        return int(value)

    @optimize
    def check_optimize(self, value):
        value = value.strip()
        if value not in self.metrics:
            raise Cfg.InvalidAttr(f'Unknown metric "{value}" provided for attribute {self.name}',
                                  f"Valid metrics are {', '.join(self.metrics)}.")


@Cfg.register_obj("hardware")
class HardwareCfg(Cfg.Obj):
    num_worker = 0
    half_precision = False
    minibatch_splits = 1
