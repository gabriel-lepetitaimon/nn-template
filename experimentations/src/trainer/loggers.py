#import mlflow
import tempfile
import shutil
from os.path import join
import os
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
import pytorch_lightning.loggers as pl_loggers
from steered_cnn.utils.attribute_dict import AttributeDict


class MlflowClientRunProxy:
    def __init__(self, mlflow_client: MlflowClient, run_id):
        self._client = mlflow_client
        self._run_id = run_id

    def log_artifact(self, path):
        self._client.log_artifact(self._run_id, path)

    def log_artifacts(self, path):
        self._client.log_artifacts(self._run_id, path)

    def log_param(self, key, value):
        self._client.log_param(self._run_id, key, value)

    def log_metric(self, key, value, timestamp=None, step=None):
        return self._client.log_metric(self._run_id, key, value, timestamp=timestamp, step=step)

    def log_text(self, text, artifact_file):
        return self._client.log_text(self._run_id, text, artifact_file)

    def log_dict(self, dictionary, artifact_file):
        return self._client.log_dict(self._run_id, dictionary, artifact_file)

    def log_batch(self, metrics=(), params=(), tags=()):
        return self._client.log_batch(self._run_id, metrics, params, tags)

    def log_figure(self, figure, artifact_file):
        return self._client.log_figure(self._run_id, figure, artifact_file)

    def log_image(self, image, artifact_file):
        return self._client.log_image(self._run_id, image, artifact_file)

    def download_artifacts(self, path, dst_path=None):
        return self._client.download_artifacts(self._run_id, path, dst_path=dst_path)

    def list_artifacts(self, path=None):
        return self._client.list_artifacts(self._run_id, path)


class Logs:
    def __init__(self):
        self.tmp = None
        self.misc = {}
        self._mlflow_logger = None

    @property
    def mlflow(self) -> MlflowClientRunProxy:
        if self._mlflow_logger is None:
            return None
        return MlflowClientRunProxy(self._mlflow_logger.experiment, self.mlflow_run_id)

    @property
    def mlflow_run_id(self):
        return self._mlflow_logger._run_id

    @property
    def loggers(self):
        return [self._mlflow_logger]

    @property
    def tmp_path(self):
        return self.tmp.name

    def setup_log(self, cfg: AttributeDict):
        cfg = AttributeDict.from_dict(cfg, True)

        EXP = cfg['experiment']['name'] if not cfg['script-arguments'].debug else 'DEBUG_RUNS'
        RUN = f"{cfg.experiment['sub-experiment']} ({cfg.experiment['sub-experiment-id']}:{cfg.trial.ID:02})"
        URI = cfg['mlflow']['uri']

        # --- SETUP MLFOW ---
        tags = cfg.experiment.tags.to_dict()
        tags['subexp'] = cfg.experiment['sub-experiment']
        tags['subexpID'] = str(cfg.experiment['sub-experiment-id'])
        tags['trial.ID'] = str(cfg.trial.ID)
        tags['trial.name'] = cfg.trial.name
        tags['trial.version'] = str(cfg.trial.version)
        tags['trial.githash'] = get_git_revision_hash()

        tags[MLFLOW_RUN_NAME] = RUN
        self._mlflow_logger = pl_loggers.MLFlowLogger(experiment_name=EXP, tracking_uri=URI, tags=tags)

        # --- CREATE TMP ---
        os.makedirs(os.path.dirname(cfg['script-arguments']['tmp-dir']), exist_ok=True)
        tmp = tempfile.TemporaryDirectory(dir=cfg['script-arguments']['tmp-dir'])
        self.tmp = tmp

        # --- SAVE CFG ---
        shutil.copy(cfg['script-arguments'].config, join(tmp.name, 'cfg.yaml'))
        self.mlflow.log_artifact(join(tmp.name, 'cfg.yaml'))
        # Sanity check of artifact saving
        artifacts = self.mlflow.list_artifacts()
        if len(artifacts) != 1 or artifacts[0].path != 'cfg.yaml':
            raise RuntimeError('The sanity check for storing artifacts failed.'
                               'Interrupting the script before the training starts.')

        exp_cfg_path = cfg.get('trial.cfg_path', None)
        if exp_cfg_path is not None:
            shutil.copy(exp_cfg_path, join(tmp.name, 'cfg_original.yaml'))
            self.mlflow.log_artifact(join(tmp.name, 'cfg_original.yaml'))

        with open(join(tmp.name, 'cfg_extended.yaml'), 'w') as f:
            cfg.to_yaml(f)

        # --- LOG PARAMS ---
        for k in cfg.model.walk():
            self.mlflow.log_param(f'model.{k}', cfg.model[k])
        for k in cfg['data-augmentation'].walk():
            self.mlflow.log_param(f'DA.{k}', cfg['data-augmentation'][k])

        self.mlflow.log_param('training.lr', cfg['hyper-parameters.lr'])
        self.mlflow.log_param('training.lr-decay', cfg['hyper-parameters.optimizer.lr-decay-factor'])
        self.mlflow.log_param('training.dropout', cfg['hyper-parameters.drop-out'])
        self.mlflow.log_param('training.batch-size', cfg['hyper-parameters.batch-size'])
        self.mlflow.log_param('training.file', cfg.training['dataset-file'])
        self.mlflow.log_param('training.dataset', cfg.training['training-dataset'])
        self.mlflow.log_param('training.seed', cfg.training['seed'])

    def log_misc(self, key, value):
        if isinstance(key, (list, tuple)):
            misc = self.misc
            for k in key[:-1]:
                if k not in misc:
                    misc[k] = {}
                misc = misc[k]
            misc[key[-1]] = value
        else:
            self.misc[key] = value

    def log_miscs(self, misc):
        self.misc.update(misc)

    def log_metric(self, metric_name, metric_value, step=None):
        return self.log_metrics({metric_name: metric_value}, step)

    def log_metrics(self, metrics, step=None):
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    def log_hyperparams(self, params):
        for logger in self.loggers:
            logger.log_hyperparams(params)

    def save_cleanup(self):
        # --- NORMALIZE CHECKPOINT FILE NAME ---
        for ckpt in os.listdir(self.tmp.name):
            if ckpt.endswith('.ckpt'):
                l = ckpt.split('-')
                if l[-1].startswith('epoch='):
                    new_ckpt = '-'.join(l[:-1]) + '.ckpt'
                    os.rename(join(self.tmp.name, ckpt), join(self.tmp.name, new_ckpt))

        # --- LOG MISC ---
        from json import dump
        with open(join(self.tmp.name, 'misc.json'), 'w') as json_file:
            dump(self.misc, json_file)

        self.mlflow.log_artifacts(self.tmp.name)

        for logger in self.loggers:
            logger.finalize()
        self.tmp.cleanup()


def get_git_revision_hash() -> str:
    import os
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=os.path.dirname(__file__)
                                   ).decode('ascii').strip()
