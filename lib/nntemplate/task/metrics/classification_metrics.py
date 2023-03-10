__all__ = ['Accuracy', 'AUROC', 'CohenKappa', 'ConfusionMatrix', 'Dice',  'F1Score', 'FBeta', 'JaccardIndex',
           'MatthewsCorrCoef', 'Precision', 'Recall', 'Specificity']

import torchmetrics as tm
import wandb
from .metrics_core import MetricCfg, Cfg, register_metric, pl
from ...misc.clip_pad import select_pixels_by_mask


class GenericClassificationMetric(MetricCfg):
    task = Cfg.oneOf('binary', 'multiclass', None, default=None)
    ignore_index = Cfg.int(None)
    validate_args = Cfg.bool(True)

    def prepare_data(self, pred, target, mask=None):
        if mask is not None:
            pred, target = select_pixels_by_mask(pred, target, mask=mask)
            return pred, target
        else:
            return pred, target

    def common_kwargs(self):
        return {}

    def create(self, num_classes):
        metric, args = self._create()
        if num_classes == 'binary':
            self['task'] = 'binary'
            num_classes = None
        task = self.task
        if task is None:
            n_classes = self.root()['task.n-classes']
            task = 'binary' if n_classes == 'binary' or n_classes <= 2 else 'multiclass'
        generic_args = {'task': task,
                        'ignore_index': self.ignore_index,
                        'validate_args': self.validate_args}
        # print('create', metric, args, self.common_kwargs(), generic_args)
        return metric(num_classes=num_classes, **args, **self.common_kwargs(), **generic_args)

    def _create(self):
        pass


class AveragableClassificationMetric(GenericClassificationMetric):
    average = Cfg.oneOf('micro', 'macro', 'weighted', 'none', None, default='macro')

    def common_kwargs(self):
        return super().common_kwargs() | {'average': self.average}


class MultilabelClassificationMetric(AveragableClassificationMetric):
    task = Cfg.oneOf('binary', 'multiclass', 'multilabel', default=None)
    num_labels = Cfg.int(None)
    top_k = Cfg.int(min=1, default=None)

    def common_kwargs(self):
        kwargs = {'num_labels': self.num_labels}
        if self.top_k:
            kwargs['top_k'] = self.top_k
        return super().common_kwargs() | kwargs


# -------------------------------------------------------------------------------------------------------
@register_metric('acc', 'classification')
class Accuracy(MultilabelClassificationMetric):
    threshold = Cfg.float(0.5)

    def log(self, module: pl.LightningModule, name: str, metric: tm.Metric):
        super().log(module, name, metric)

    def _create(self):
        return tm.Accuracy, {'threshold': self.threshold}


@register_metric('auroc', 'classification')
class AUROC(MultilabelClassificationMetric):
    average = Cfg.oneOf('macro', 'weighted', 'none', None, default='macro')
    max_fpr = Cfg.float(None, min=0)

    def _create(self):
        return tm.AUROC, {'max_fpr': self.max_fpr, 'compute_on_cpu': True}


@register_metric('kappa', 'classification')
class CohenKappa(GenericClassificationMetric):
    threshold = Cfg.float(0.5)
    weights = Cfg.oneOf('linear', 'quadratic', default=None)

    def _create(self):
        return tm.CohenKappa, {'threshold': self.threshold, 'weights': self.weights}


@register_metric('confmat', 'classification')
class ConfusionMatrix(AveragableClassificationMetric):
    average = None
    threshold = Cfg.float(0.5)
    normalize = Cfg.oneOf('true', 'pred', 'all', default=None)

    def _create(self):
        return tm.ConfusionMatrix, {'threshold': self.threshold, 'normalize': self.normalize}

    def log(self, module: pl.LightningModule, name: str, metric: tm.Metric):
        if name.startswith('val'):
            return
        from torchmetrics.classification.confusion_matrix import BinaryConfusionMatrix
        n_classes = 2 if isinstance(metric, BinaryConfusionMatrix) else metric.num_classes

        task = self.root()['task']
        class_names = getattr(task, 'classes', None)
        if class_names is None or len(class_names) != n_classes:
            class_names = [f"Class {i}" for i in range(n_classes)]

        confmat_data = []
        for i in range(n_classes):
            for j in range(n_classes):
                confmat_data.append([class_names[i], class_names[j], metric.confmat[i, j].detach().cpu().item()])

        confmat = wandb.plot_table(
            "wandb/confusion_matrix/v1",
            wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=confmat_data),
            {
                "Actual": "Actual",
                "Predicted": "Predicted",
                "nPredictions": "nPredictions",
            },
        )
        wandb.log({name: confmat})


@register_metric('dice', 'classification')
class Dice(MultilabelClassificationMetric):
    average = Cfg.oneOf('micro', 'macro', 'none', None, 'samples', default='macro')
    mdmc_average = Cfg.oneOf('samplewise', 'global', default=None)
    threshold = Cfg.float(0.5)

    def _create(self):
        return tm.Dice, {'threshold': self.threshold, 'mdmc_average': self.mdmc_average}

    def log(self, module: pl.LightningModule, name: str, metric: tm.Metric):
        super().log(module, name, metric)
        # print('Dice | ', {k: v for k, v in zip('tp,fp,tn,fn'.split(','), metric._get_final_stats())}, metric.compute())


@register_metric('f1', 'classification')
class F1Score(MultilabelClassificationMetric):
    threshold = Cfg.float(0.5)

    def _create(self):
        return tm.F1Score, {'threshold': self.threshold}


@register_metric('fBeta', 'classification')
class FBeta(MultilabelClassificationMetric):
    threshold = Cfg.float(0.5)
    beta = Cfg.float(1)

    def _create(self):
        return tm.FBetaScore, {'threshold': self.threshold, 'beta': self.beta}


@register_metric('iou', 'classification')
class JaccardIndex(MultilabelClassificationMetric):
    threshold = Cfg.float(0.5)
    top_k = None

    def _create(self):
        return tm.JaccardIndex, {'threshold': self.threshold}


@register_metric('mcc', 'classification')
class MatthewsCorrCoef(MultilabelClassificationMetric):
    threshold = Cfg.float(0.5)
    top_k = None

    def _create(self):
        return tm.MatthewsCorrCoef, {'threshold': self.threshold}


@register_metric('precision', 'classification')
class Precision(MultilabelClassificationMetric):
    threshold = Cfg.float(0.5)

    def _create(self):
        return tm.Precision, {'threshold': self.threshold}


@register_metric('recall', 'classification')
class Recall(MultilabelClassificationMetric):
    threshold = Cfg.float(0.5)

    def _create(self):
        return tm.Recall, {'threshold': self.threshold}


@register_metric('specificity', 'classification')
class Specificity(MultilabelClassificationMetric):
    threshold = Cfg.float(0.5)

    def _create(self):
        return tm.Specificity, {'threshold': self.threshold}
