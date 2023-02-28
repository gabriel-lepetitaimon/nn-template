import tempfile
import torch
import wandb
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
    log_model = Cfg.oneOf(True, False, 'all', default=True)

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

    def setup_logs(self, wandb_init_kwargs) -> WandbLogger:
        from .hardware import HardwareCfg
        from .hyperparameters_tuning.optuna import OptunaCfg

        experiment: ExperimentCfg = self.root()['experiment']
        hardware: HardwareCfg = self.root()['hardware']

        project = experiment.project if not hardware.debug else 'Debug & Test'
        top_config = {'EXP': experiment.name, 'ID': experiment.run_id}
        settings = wandb.Settings(launch=False)
        if 'settings' in wandb_init_kwargs:
            settings.update(wandb_init_kwargs.pop('settings'))

        logger = WandbLogger(name=experiment.run_name,
                             project=project,
                             tags=experiment.tags,
                             config=top_config | self.root().to_dict(exportable=True),
                             group=self.group,
                             job_type=self.job_type,
                             entity=self.entity,
                             save_dir=self.dir,
                             notes=self.notes,
                             log_model=self.log_model,
                             settings=settings,
                             **wandb_init_kwargs
                             )

        # Log custom job artifacts if launch was not forced to True.
        if not settings['launch']:
            #   --- Job Artifact ---
            job_art_name = 'EXP-'+experiment.name.replace(' ', '_')
            if experiment.run_id == 1 or experiment.run_id is None:
                job_artifact = wandb.Artifact(job_art_name, type='job')

                # Log configuration files
                if self.root().mark:
                    parser = self.root().mark.parser
                    if parser:
                        job_artifact.add_file(parser.files[0].path, 'main.yaml')
                        for cfg_file in parser.files[1:]:
                            job_artifact.add_file(cfg_file.path)

                logger._experiment.use_artifact(job_artifact)
            else:
                logger.use_artifact(job_art_name+':latest')

            #   --- Job Artifact ---
            if experiment.run_id is not None:
                run_art_name = 'RUN-'+experiment.run_name.replace(' ', '_')
                run_artifact = wandb.Artifact(run_art_name, type='hyper-parameters')

                # Log hyperparameters value
                optuna_cfg: OptunaCfg = self.root().get('optuna', None)
                if isinstance(optuna_cfg, OptunaCfg):
                    with tempfile.NamedTemporaryFile('w+') as fp:
                        optuna_cfg.hyper_parameters().to_yaml(fp)
                        run_artifact.add_file(fp.name, 'hyper-parameters.yaml')
                    with tempfile.NamedTemporaryFile('w+') as fp:
                        self.root().to_yaml(fp)
                        run_artifact.add_file(fp.name, 'run_config.yaml')
                logger._experiment.use_artifact(run_artifact)

        return logger


class WandBLogContext:
    def __init__(self, cfg: WandBCfg, wandb_init_kwargs=None):
        self.cfg = cfg
        self.wandb_init_kwargs = {} if wandb_init_kwargs is None else wandb_init_kwargs
        self.logger = None

    def __enter__(self):
        from .hyperparameters_tuning.optuna import OptunaCfg
        experiment: ExperimentCfg = self.cfg.root()['experiment']

        self.logger = self.cfg.setup_logs(self.wandb_init_kwargs)

        print(f"========== STARTING RUN: {experiment.run_name} =============")
        optuna_cfg: OptunaCfg = self.cfg.root().get('optuna', None)
        if isinstance(optuna_cfg, OptunaCfg):
            print('\t *** Hyper-Parameters ***')
            print('\t' + optuna_cfg.hyper_parameters().to_yaml().replace('\n', '\n\t'))

        self.cfg._wandb_log_context = self
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        from optuna import TrialPruned
        if isinstance(exc_val, TrialPruned):
            print(" ========= RUN PRUNED ===========")
            self.logger.finalize(status="aborted")
            exit_code = 1
        elif exc_type is not None:
            print(" ========= RUN CRASHED ===========")
            self.logger.finalize(status="failed")
            exit_code = 10
        else:
            print(" ========= RUN SUCCESS ===========")
            self.logger.finalize(status="success")
            exit_code = 0
        wandb.finish(exit_code)

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
        if self.run_id is None:
            return self.name
        return self.name + f'-{self.run_id:02d}'

    @cached_property
    def experiment_hash(self) -> str:
        from json import dumps
        import hashlib
        cfg = self.root().to_dict(exportable=True)
        cfg['experiment'].pop('run_id', 0)
        return hashlib.md5(dumps(cfg).encode()).hexdigest()
