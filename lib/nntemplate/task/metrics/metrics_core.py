__all__ = ['MetricCfg', 'Metric', 'metric_attr', 'metrics_attr', 'register_metric', 'MonitoredMetricCfg', 'check_metric_name']

from typing import Dict, Iterable
import pytorch_lightning as pl
import torchmetrics as tm
from torchmetrics import Metric

from ...config import Cfg
from ...config.cfg_dict import UNDEFINED


class MetricCfg(Cfg.Obj):
    metric_type = "metric"
    metric_name = "METRIC_NAME"

    def prepare_data(self, *args):
        return args

    def create(self, **kwargs):
        pass

    def log(self, module: pl.LightningModule, name: str, metric: tm.Metric):
        module.log(name, metric, add_dataloader_idx=False, enable_graph=False)


_METRICS: Dict[str, MetricCfg] = {}


def _metrics_by_type(metric_type):
    if isinstance(metric_type, str):
        metric_type = [m.strip() for m in metric_type.split(',') if m.strip()]
    if metric_type:
        metrics = {k: m for k, m in _METRICS.items() if m.metric_type in metric_type}
        if not metrics:
            raise ValueError(f'Unkown metric type: {metric_type}. Valid types are '
                             f'{",".join(set(m.metric_type for m in _METRICS.values()))}.')
    return _METRICS


def metric_attr(default=UNDEFINED, metric_type=None, nullable=None):
    return Cfg.obj(default=default, obj_types=_metrics_by_type(metric_type), nullable=nullable)


def metrics_attr(default=UNDEFINED, metric_type: Iterable[str] | str = None):
    return Cfg.obj_list(main_key='type', obj_types=_metrics_by_type(metric_type),
                        type_key='type', default=default)


def register_metric(name: str, metric_type: str = 'metric'):
    def register(f_metric: MetricCfg):
        f_metric.metric_type = metric_type
        f_metric.metric_name = name
        _METRICS[name] = f_metric
        return f_metric

    return register


class MonitoredMetricCfg(Cfg.Obj):
    metric = Cfg.str()
    mode = Cfg.oneOf('min', 'max', default='max')

    @metric.post_checker
    def check_metric(self, metric):
        if metric.endswith('^'):
            metric = metric[:-1].strip()
            self['mode'] = 'min'
        check_metric_name(self, metric, 'checkpoint')
        if not metric.startswith(('val', 'train')):
            raise Cfg.InvalidAttr(f'Invalid metric: "{metric}"',
                                  'Monitored metrics during training must be computed on '
                                  'the validation or training datasets.')
        return metric


__cached_valid_metrics_names: tuple[str] | None = None


def check_metric_name(cfg: Cfg.Dict, metric_name: str, attr_name: str):
    from ..task import LightningTaskCfg
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
