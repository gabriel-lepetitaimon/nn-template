from .loss_core import LossCfg, register_loss, Cfg
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss as CELoss
from nntemplate.utils.torch import select_pixels_by_mask


@register_loss('cross-entropy')
class CrossEntropyLoss(LossCfg):
    with_logits = Cfg.bool(True)

    def loss(self):
        if self.root().get('task.n-classes', None) == 'binary':
            if self.with_logits:
                loss = BCEWithLogitsLoss()
            else:
                loss = BCELoss()
        else:
            loss = CELoss()
        self._loss_fn = loss
        return self._compute_loss

    def _compute_loss(self, pred, target, mask):
        pred, target = select_pixels_by_mask(pred, target, mask=mask)
        return self._loss_fn(pred, target)
