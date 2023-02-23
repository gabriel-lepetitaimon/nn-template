from typing import Dict, Iterable
import pytorch_lightning as pl
import torchmetrics as tm

from ...config import Cfg
from ...config.cfg_dict import UNDEFINED


class Metric(Cfg.Obj):
    metric_type = "metric"
    metric_name = "METRIC_NAME"

    def prepare_data(self, *args):
        return args

    def create(self, **kwargs):
        pass

    def log(self, module: pl.LightningModule, name: str, metric: tm.Metric):
        module.log(name, metric, add_dataloader_idx=False, enable_graph=False)


_METRICS: Dict[str, Metric] = {}


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
    def register(f_metric: Metric):
        f_metric.metric_type = metric_type
        f_metric.metric_name = name
        _METRICS[name] = f_metric
        return f_metric

    return register
