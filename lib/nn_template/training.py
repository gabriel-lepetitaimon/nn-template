import pytorch_lightning as pl
import math

from . import Cfg


@Cfg.register_obj("training")
class TrainingCfg(Cfg.Obj):
    seed = 1234
    minibatch = Cfg.int(None)
    max_epoch = Cfg.int()

    validate_every_n_epoch = Cfg.float(min=0)
    objective = Cfg.str('val-acc')
    monitor = Cfg.strList(default=None)
    direction = Cfg.oneOf('max', 'min', default='max')
    n_runs = Cfg.int(min=0)

    checkpoint_metrics = Cfg.strList([])

    def _init_after_populate(self):
        # Check objective
        self.check_checkpoint_metrics(self.checkpoint_metrics)
        self.check_objective(self.objective)

    @objective.post_checker
    def check_objective(self, value):
        valid_metric_names = self.root()['task'].metrics_names
        if valid_metric_names is not None:
            value = value.strip()
            if value not in valid_metric_names:
                raise Cfg.InvalidAttr(f'Unknown metric "{value}" provided for attribute {self.name}.objective',
                                      f"Valid metrics are {', '.join(valid_metric_names)}")
        return value

    @checkpoint_metrics.post_checker
    def check_checkpoint_metrics(self, value):
        valid_metric_names = self.root()['task'].metrics_names
        if valid_metric_names is not None:
            for metric in value:
                name = metric[:-1].strip() if metric.endswith('!') else metric
                if name not in valid_metric_names:
                    raise Cfg.InvalidAttr(f'Unknown metric "{name}" provided for attribute {self.name}.checkpoint-metrics',
                                          f"Valid metrics are {', '.join(valid_metric_names)}")
        return value

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

    def create_trainer(self, callbacks, **trainer_kwargs) -> pl.Trainer:
        from .experiment import ExperimentCfg
        from .hardware import HardwareCfg
        from pytorch_lightning.callbacks import ModelCheckpoint
        hardware: HardwareCfg = self.root()['hardware']
        # experiment: ExperimentCfg = self.root()['experiment']

        for metric in self.checkpoint_metrics:
            if metric.endswith('!'):
                metric = metric[:-1].strip()
                mode='min'
            else:
                metric = metric.strip()
                mode = 'max'
            callbacks += [ModelCheckpoint(monitor=metric, mode=mode)]

        kwargs = dict(gpus=hardware.gpus, callbacks=callbacks,
                      max_epochs=self.max_epoch, check_val_every_n_epoch=self.validate_every_n_epoch,
                      accumulate_grad_batches=hardware.minibatch_splits,
                      progress_bar_refresh_rate=1 if hardware.debug else 0
                      )

        if hardware.precision:
            kwargs['precision'] = hardware.precision

        kwargs.update(trainer_kwargs)
        return pl.Trainer(**kwargs)
