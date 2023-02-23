from .loss_core import Loss, register_loss, Cfg
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss as CELoss


@register_loss('cross-entropy')
class CrossEntropyLoss(Loss):
    with_logits = Cfg.bool(True)

    def create(self):
        if self.root().get('task.n-classes', None) == 'binary':
            if self.with_logits:
                return BCEWithLogitsLoss()
            else:
                return BCELoss()
        else:
            return CELoss(ignore_index=-1)
