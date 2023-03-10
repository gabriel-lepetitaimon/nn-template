from functools import cached_property
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import tempfile
import wandb

from .config import Cfg


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
        self.api = wandb.Api()
        wandb_init_args = dict(resume=resume, reinit=True)
        return WandBLogContext(self, wandb_init_args)

    def pl_callbacks(self) -> list[pl.Callback]:
        return [LogCallback(self)]

    def setup_logs(self, wandb_init_kwargs) -> WandbLogger:
        from .hardware import HardwareCfg
        from .hyperparameters_tuning.optuna import OptunaCfg

        experiment: ExperimentCfg = self.root()['experiment']
        hardware: HardwareCfg = self.root()['hardware']

        # Fix run_id
        versions = [v for v in self.root().current_versions() if 'experiment.name' not in v.versions]
        if len(versions):
            ids = (v.version_id for v in versions)
            strides = np.cumprod([1] + [len(v.versions) for v in versions])
            experiment['run_id'] = sum(id * stride for id, stride in zip(ids, strides))

        # Init Wandb
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
                             log_model=False,   # Model are logged manually at the end of training
                             settings=settings,
                             **wandb_init_kwargs
                             )

        # Log custom job artifacts if launch was not forced to True.
        if not settings['launch']:
            #   --- Job Artifact ---
            job_art_name = experiment.name.replace(' ', '_')
            job_artifact = None
            try:
                job_versions = list(self.api.artifact_versions('job', experiment.project + '/' + job_art_name))
            except wandb.CommError:
                job_versions = []
            if job_versions:
                for v in job_versions:
                    if v.metadata['exp_hash'] == experiment.experiment_hash:
                        job_artifact = logger.use_artifact(experiment.project+'/'+v.name)
                        break
            if job_artifact is None:
                job_artifact = wandb.Artifact(job_art_name, type='job')
                job_artifact.metadata['exp_hash'] = experiment.experiment_hash

                # Log configuration files
                if self.root().mark:
                    parser: Cfg.Parser = self.root().mark.parser
                    if parser:
                        job_artifact.add_file(parser.files[0].path, 'main.yaml')
                        for cfg_file in parser.files[1:]:
                            job_artifact.add_file(cfg_file.path)

                logger._experiment.use_artifact(job_artifact)

            #   --- Run Artifact ---
            if experiment.run_id is not None:
                run_art_name = experiment.run_name.replace(' ', '_')
                run_artifact = wandb.Artifact(run_art_name, type='hyper-parameters')

                # Log hyperparameters value
                optuna_cfg: OptunaCfg = self.root().get('optuna', None)
                if isinstance(optuna_cfg, OptunaCfg):
                    with tempfile.NamedTemporaryFile('w+') as fp:
                        hps = optuna_cfg.hyper_parameters().copy()
                        for version in versions:
                            hps.update(version)
                        hps.to_yaml(fp)
                        run_artifact.add_file(fp.name, 'hyper-parameters.yaml')
                    with tempfile.NamedTemporaryFile('w+') as fp:
                        self.root().to_yaml(fp)
                        run_artifact.add_file(fp.name, 'run_config.yaml')
                logger._experiment.use_artifact(run_artifact)

        return logger

    def save_logs(self):

        if self.log_model:
            # Save models
            from .training import TrainingCfg
            training_cfg: TrainingCfg = self.root()['training']
            experiment: ExperimentCfg = self.root()['experiment']
            model_art_name = experiment.name.replace(' ', '_')+'.models'
            model_art = wandb.Artifact(model_art_name, type='model')

            try:
                versions = list(self.api.artifact_versions('model', experiment.project+'/'+model_art_name))
            except wandb.CommError:
                versions = []
            alias = ['latest']

            for ckpg in training_cfg.checkpoints.list():
                metric = ckpg.metric
                model_art.add_file(ckpg.checkpoint.best_model_path, 'best-'+metric+'.ckpg')
                model_art.metadata[metric] = ckpg.checkpoint.best_model_score

                if not versions or all(v.metadata[metric] <= ckpg.checkpoint.best_model_score for v in versions):
                    alias += ['best-'+metric]

            self.logger.experiment.log_artifact(model_art, aliases=alias)


class LogCallback(pl.Callback):
    def __init__(self, cfg: WandBCfg):
        self.cfg = cfg

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.cfg.watch.log:
            self.cfg.logger.watch(pl_module, log=self.cfg.watch.log, log_freq=self.cfg.watch.log_freq,
                                  log_graph=self.cfg.watch.log_graph)


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
            self.cfg.save_logs()
            self.logger.finalize(status="success")
            exit_code = 0
        wandb.finish(exit_code)

        self.cfg._wandb_log_context = None


@Cfg.register_obj('experiment')
class ExperimentCfg(Cfg.Obj):
    name = Cfg.str()
    project = Cfg.str(None)
    run_id = Cfg.int(None)
    trial_id = Cfg.int(None)
    tags = Cfg.Dict()

    wandb: WandBCfg = Cfg.obj()

    @property
    def run_name(self):
        if self.trial_id is not None:
            if self.run_id is not None:
                return self.name + f'-{self.trial_id:02d}.{self.run_id:02d}'
            else:
                return self.name + f'-{self.trial_id:02d}'
        elif self.run_id is not None:
            return self.name + f'-{self.run_id:02d}'
        else:
            return self.name

    @cached_property
    def experiment_hash(self) -> str:
        from json import dumps
        import hashlib
        cfg = self.root().to_dict(exportable=True)
        cfg['experiment'].pop('run_id', 0)
        return hashlib.md5(dumps(cfg).encode()).hexdigest()
