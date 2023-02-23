import torch
from pytorch_lightning.loggers import WandbLogger
from functools import cached_property

from .config.config import Cfg


class WandBWatchCfg(Cfg.Obj):
    log = Cfg.oneOf('gradients', 'parameters', 'all', False, default=False)
    log_freq = Cfg.int(100)
    log_graph = Cfg.bool(True)


class WandBCfg(Cfg.Obj):
    entity = Cfg.str(None)
    group = Cfg.str(None)
    job_type = Cfg.str(None)
    notes = Cfg.str(None)
    dir = Cfg.str(None)

    watch: WandBWatchCfg = Cfg.obj(shortcut='log', default=False)
    log_model = Cfg.oneOf(True, False, 'all', default=False)

    @property
    def logger(self) -> WandbLogger | None:
        context = getattr(self, '_wandb_log_context', None)
        return context.logger if context is not None else None

    def init_logs(self, resume="never"):
        wandb_init_args = dict(resume=resume, reinit=True)
        return WandBLogContext(self, wandb_init_args)

    def setup_model_log(self, model: torch.nn.Module):
        if self.watch.log:
            self.logger.watch(model, log=self.watch.log, log_freq=self.watch.log_freq, log_graph=self.watch.log_graph)


class WandBLogContext:
    def __init__(self, cfg: WandBCfg, wandb_init_kwargs=None):
        self.cfg = cfg
        self.wandb_init_kwargs = {} if wandb_init_kwargs is None else wandb_init_kwargs
        self.logger = None

    def __enter__(self):
        from .hardware import HardwareCfg
        wandb_cfg: WandBCfg = self.cfg
        experiment: ExperimentCfg = self.cfg.root()['experiment']
        hardware: HardwareCfg = self.cfg.root()['hardware']

        project = experiment.project if not hardware.debug else 'Debug & Test'
        top_config = {'ID': experiment.run_id}

        print(f"========== STARTING RUN {experiment.run_id} =============")
        self.logger = WandbLogger(name=experiment.name,
                                  project=project,
                                  tags=experiment.tags,
                                  config=top_config | self.cfg.root().to_dict(exportable=True),
                                  group=wandb_cfg.group,
                                  job_type=wandb_cfg.job_type,
                                  entity=wandb_cfg.entity,
                                  save_dir=wandb_cfg.dir,
                                  notes=wandb_cfg.notes,
                                  log_model=wandb_cfg.log_model,
                                  **self.wandb_init_kwargs
                                  )
        self.cfg._wandb_log_context = self
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        from optuna import TrialPruned
        if isinstance(exc_type, TrialPruned):
            print(" ========= RUN PRUNED ===========")
            self.logger.finalize(status="aborted")
        elif exc_type is not None:
            print(" ========= RUN CRASHED ===========")
            self.logger.finalize(status="failed")
        else:
            print(" ========= RUN SUCCESS ===========")
            self.logger.finalize(status="success")

        self.cfg._wandb_log_context = None


@Cfg.register_obj('experiment')
class ExperimentCfg(Cfg.Obj):
    name = Cfg.str()
    project = Cfg.str(None)
    run_id = Cfg.int(None)
    tags = Cfg.Dict()

    wandb: WandBCfg = Cfg.obj()

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
