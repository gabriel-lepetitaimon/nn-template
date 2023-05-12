from ..experiment import ExperimentCfg
from ..task import LightningTaskCfg
from .. import Cfg

def load_model_from_cfg(cfg, best_metric=None):
    from torch import load
    import wandb

    cfg = Cfg.load(cfg)
    exp: ExperimentCfg = cfg.root()['experiment']
    task_cfg: LightningTaskCfg = cfg.root()['task']

    model = cfg.model.create()
    task = task_cfg.create(model)

    alias = 'best'
    if best_metric:
        alias += '-' + best_metric

    api = wandb.Api()
    model_name = exp.name.replace(' ', '_')+'.models:'+alias
    model_artifact = api.artifact(model_name, project=exp.project)
    model_artifact.download()

    model.load(model_artifact.get_path(alias+'.ckpg'))


def load_model_from_wandb(project, name, best_metric=None):
    import wandb
    api = wandb.Api()
    artifact = api.artifact(name, project=project)
    artifact.download()
    return artifact