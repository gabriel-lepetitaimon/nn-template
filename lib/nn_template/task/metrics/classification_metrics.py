import torchmetrics as tm
import wandb
from .metrics_core import Metrics, Cfg, register_metric, pl


class GenericClassificationMetric(Metrics):
    task = Cfg.oneOf('binary', 'multiclass', default='multiclass')
    ignore_index = Cfg.int(None)
    validate_args = Cfg.bool(True)

    def common_kwargs(self):
        return {}

    def create(self, num_classes):
        metric, args = self._create()
        if num_classes == 'binary':
            self.task = 'binary'
            num_classes = None
        generic_args = {'task': self.task,
                        'ignore_index': self.ignore_index,
                        'validate_args': self.validate_args}
        return metric(num_classes=num_classes, **args, **self.common_kwargs(), **generic_args)

    def _create(self):
        pass


class AveragableClassificationMetric(GenericClassificationMetric):
    average = Cfg.oneOf('micro', 'macro', 'weighted', 'none', None, default='micro')

    def common_kwargs(self):
        return super().common_kwargs() | {'average': self.average}


class MultilabelClassificationMetric(AveragableClassificationMetric):
    task = Cfg.oneOf('binary', 'multiclass', 'multilabel', default='multiclass')
    num_labels = Cfg.int(None)
    top_k = Cfg.int(1)

    def common_kwargs(self):
        return super().common_kwargs() | \
            {'num_labels': self.num_labels, 'top_k': self.top_k}


# -------------------------------------------------------------------------------------------------------
@register_metric('acc', 'classification')
class Accuracy(MultilabelClassificationMetric):
    threshold = Cfg.float(0.5)

    def _create(self):
        return tm.Accuracy, {'threshold': self.threshold}


@register_metric('auroc', 'classification')
class AUROC(AveragableClassificationMetric):
    average = Cfg.oneOf('macro', 'weighted', 'none', None, default='macro')
    max_fpr = Cfg.float(None, min=0)

    def _create(self):
        return tm.AUROC, {'max_fpr': self.max_fpr}


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
    class_names = Cfg.strList(None)

    def _create(self):
        return tm.ConfusionMatrix, {'threshold': self.threshold, 'normalize': self.normalize}

    def log(self, trainer: pl.LightningModule, name: str, metric: tm.Metric):
        from torchmetrics.classification.confusion_matrix import BinaryConfusionMatrix
        n_classes = 2 if isinstance(metric, BinaryConfusionMatrix) else metric.num_classes

        class_names = self.class_names
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
        wandb.log(name, confmat)


@register_metric('dice', 'classification')
class Dice(MultilabelClassificationMetric):
    average = Cfg.oneOf('micro', 'macro', 'weighted', 'none', None, 'samples', default='micro')
    mdmc_average = Cfg.oneOf('samplewise', 'global', default=None)
    threshold = Cfg.float(0.5)

    def _create(self):
        return tm.Dice, {'threshold': self.threshold, 'mdmc_average': self.mdmc_average}


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
