from .loss_core import Loss, register_loss, Cfg
import torch.nn.functional as F


@register_loss('cross-entropy')
class CrossEntropyLoss(Loss):
    with_logits = Cfg.bool(True)

    def create(self, n_classes):
        if n_classes == 'binary':
            if self.with_logits:
                def binary_cross_entropy(pred, target):
                    return F.binary_cross_entropy_with_logits(pred, target.float())
            else:
                def binary_cross_entropy(pred, target):
                    return F.binary_cross_entropy(pred, target.float())
            return binary_cross_entropy
        else:
            def cross_entropy(pred, target):
                return F.cross_entropy(pred, target.float())
            return cross_entropy