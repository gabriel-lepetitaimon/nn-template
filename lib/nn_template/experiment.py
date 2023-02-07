import torch
from functools import cached_property

from .config.config import Cfg


class WandBWatchCfg(Cfg.Obj):
    log = Cfg.oneOf('gradients', 'parameters', 'all', None, default=None)
    log_freq = Cfg.int(100)
    log_graph = Cfg.bool(True)


class WandBCfg(Cfg.Obj):
    entity = Cfg.str(None)
    group = Cfg.str(None)
    job_type = Cfg.str(None)
    notes = Cfg.str(None)
    dir = Cfg.str(None)

    watch: WandBWatchCfg = Cfg.obj(shortcut='log', default=None)
    log_model = Cfg.oneOf(True, False, 'all', default=False)

    @cached_property
    def logger(self):
        from pytorch_lightning.loggers import WandbLogger
        project = self.project if not self.root().get('hardware.debug', False) else 'DEBUG'
        return WandbLogger(name=self.parent.run_name,
                           id=str(self.parent.run_id),
                           project=project,
                           tags=self.parent.tags,
                           config=self.root().to_dict(exportable=True),
                           group=self.group,
                           job_type=self.job_type,
                           entity=self.entity,
                           save_dir=self.dir,
                           notes=self.notes,
                           log_model=self.log_model
                           )

    def setup_model_log(self, model: torch.nn.Module):
        self.logger.watch(model, log=self.watch.log, log_freq=self.watch.log_freq, log_graph=self.watch.log_graph)


@Cfg.register_obj('experiment')
class ExperimentCfg(Cfg.Obj):
    name = Cfg.str()
    project = Cfg.str(None)
    run_id = Cfg.int(None)
    tags = Cfg.strList([])

    wandb: WandBCfg = Cfg.Obj()

    @property
    def run_name(self):
        if self.run_id is not None:
            return self.name
        return self.name

    @cached_property
    def experiment_hash(self) -> str:
        from json import dumps
        import hashlib
        cfg = self.root().to_dict(exportable=True)
        cfg['experiment'].pop('run_id', 0)
        return hashlib.md5(dumps(cfg).encode()).hexdigest()
