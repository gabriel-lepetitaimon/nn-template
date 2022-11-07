from .. import Cfg
import math


class LossCfg(Cfg.Obj):
    name = Cfg.oneOf('cross-entropy', 'dice')

    def create_loss(self):
        n_class = self.root().get('task.n-classes', 'binary')
        pass


@Cfg.register_obj("training")
class TrainingCfg(Cfg.Obj):
    seed = 1234
    minibatch = Cfg.int(None)
    max_epoch = Cfg.int()

    loss: LossCfg = Cfg.obj(shortcut='name')

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

    @property
    def minibatch_size(self):
        return math.ceil(self.minibatch / self.root().get('hardware.minibatch-splits', 1))

    def lightning_args(self):
        args = dict()
        if self.minibatch is None:
            args['auto_scale_batch_size'] = True
        return args