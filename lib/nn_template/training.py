from functools import cached_property
import math
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import wandb

from . import Cfg

from .task import LightningTaskCfg
from .hardware import HardwareCfg
from .datasets import DatasetsCfg


class CheckpointCfg(Cfg.Obj):
    metric = Cfg.str()
    mode = Cfg.oneOf('min', 'max', default='max')

    @metric.post_checker
    def check_metric(self, metric):
        if metric.endswith('^'):
            metric = metric[:-1].strip()
            self['mode'] = 'min'
        check_metric_name(self, metric, 'checkpoint')
        if not metric.startswith(('val', 'train')):
            raise Cfg.InvalidAttr(f'Invalid metric for checkpoint: "{metric}"',
                                  'Metrics monitored by checkpoint must be computed on '
                                  'the validation or training datasets.')
        return metric

    def create(self) -> ModelCheckpoint:
        self._checkpoint = ModelCheckpoint(monitor=self.metric, mode=self.mode)
        return self._checkpoint

    @property
    def checkpoint(self) -> ModelCheckpoint:
        if not hasattr(self, '_checkpoint'):
            return self.create()
        return self._checkpoint

    @property
    def best_epoch(self):
        best_path = self.checkpoint.best_model_path
        if best_path:
            return int(best_path[:-5].rsplit('-', 1)[1][6:])
        return -1


@Cfg.register_obj("training")
class TrainingCfg(Cfg.Obj):
    seed = 1234
    minibatch = Cfg.int(None)
    gradient_clip = 0
    max_epoch = Cfg.int()

    checkpoints = Cfg.obj_list(main_key='metric', obj_types=CheckpointCfg)
    validate_every_n_epoch = Cfg.float(min=0)
    objective = Cfg.str('val-acc')
    monitor = Cfg.strList(default=None)
    direction = Cfg.oneOf('max', 'min', default='max')
    n_runs = Cfg.int(min=0)

    def _init_after_populate(self):
        self.check_objective(self.objective)

    @objective.post_checker
    def check_objective(self, value):
        check_metric_name(self, value, 'objective')
        try:
            if not any(value == ckpt.metric for ckpt in self.checkpoints.values()):
                raise Cfg.InvalidAttr(f'Invalid objective metric "{value}"',
                                      f'The objective metric must be monitored by a checkpoint.')
        except AttributeError:
            pass
        return value

    def configure_seed(self):
        import os

        if self.seed == 0:
            self.seed = int.from_bytes(os.getrandom(32), 'little', signed=False)

        seed = self.seed
        pl.seed_everything(seed)

    @property
    def minibatch_size(self):
        return math.ceil(self.minibatch / self.root().get('hardware.minibatch-splits', 1))

    def lightning_args(self):
        args = dict()
        if self.minibatch is None:
            args['auto_scale_batch_size'] = True
        return args

    def _hardware_args(self):
        hardware: HardwareCfg = self.root()['hardware']
        return dict(gpus=hardware.gpus,
                    progress_bar_refresh_rate=1 if hardware.debug else 0,
                    fast_dev_run=10 if hardware.debug == 'fast' else None)

    def create_trainer(self, callbacks, setup_dataset=True, **trainer_kwargs) -> pl.Trainer:
        hardware: HardwareCfg = self.root()['hardware']

        for name, checkpoint in self.checkpoints.items():
            callbacks += [checkpoint.create()]

        kwargs = dict(callbacks=callbacks,
                      max_epochs=self.max_epoch, check_val_every_n_epoch=self.validate_every_n_epoch,
                      accumulate_grad_batches=hardware.minibatch_splits, gradient_clip_val=self.gradient_clip)
        if hardware.precision:
            kwargs['precision'] = hardware.precision

        trainer = pl.Trainer(** self._hardware_args() | kwargs | trainer_kwargs)

        if setup_dataset:
            datasets_cfg: DatasetsCfg = self.root()['datasets']
            trainer.train_dataloader = DataLoader(datasets_cfg.train.dataset(),
                                                  pin_memory=True,
                                                  shuffle=True,
                                                  batch_size=self.minibatch_size,
                                                  num_workers=hardware.num_worker)
            trainer.val_dataloaders = DataLoader(datasets_cfg.validate.dataset(),
                                                 pin_memory=True,
                                                 num_workers=6,
                                                 batch_size=6)
        return trainer

    def create_tester(self, callbacks, setup_datasets=True, **tester_kwargs) -> pl.Trainer:
        kwargs = dict(callbacks=callbacks,)

        tester = pl.Trainer(**self._hardware_args() | kwargs | tester_kwargs)

        if setup_datasets:
            datasets_cfg: DatasetsCfg = self.root()['datasets']
            test_datasets = {name: DataLoader(cfg.dataset(),
                                              pin_memory=True,
                                              batch_size=6, num_workers=6)
                             for name, cfg in datasets_cfg.items()}
            tester.test_dataloaders = list(test_datasets.values())
            tester.test_dataloaders_names = list(test_datasets.keys())

    def log_best_ckpt_metrics(self):
        best_metric = {'best-'+k: v.checkpoint.best_model_score for k, v in self.checkpoints.items()}
        best_epoch = {'best-'+k+'-epoch': ckpt.best_epoch for k, ckpt in self.checkpoints.items()}
        wandb.log(best_metric | best_epoch)

    @cached_property
    def objective_ckpt_cfg(self) -> CheckpointCfg:
        for ckpt in self.checkpoints.values():
            if ckpt.metric == self.objective:
                return ckpt

    @property
    def objective_best_ckpt_path(self):
        return self.objective_ckpt_cfg.checkpoint.best_model_path

    @property
    def objective_best_value(self):
        return self.objective_ckpt_cfg.checkpoint.best_model_score


__cached_valid_metrics_names: tuple[str] | None = None


def check_metric_name(cfg: Cfg.Dict, metric_name: str, attr_name: str):
    global __cached_valid_metrics_names
    if __cached_valid_metrics_names is None:
        task: LightningTaskCfg = cfg.root()['task']
        if isinstance(task, LightningTaskCfg):
            __cached_valid_metrics_names = task.metrics_names

    if __cached_valid_metrics_names is not None:
        metric_name = metric_name.strip()
        if metric_name not in __cached_valid_metrics_names:
            raise Cfg.InvalidAttr(f'Unknown metric "{metric_name}" provided for attribute {cfg.fullname}.{attr_name}',
                                  f"Valid metrics are {', '.join(__cached_valid_metrics_names)}")
    return metric_name
