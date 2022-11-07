from .. import Cfg
import torch
import torch.nn.functional as F

# TODO: multiple obj_types for one Cfg.obj()

class DiceLoss(Cfg.Obj):
    eps = 1e-6

    def create(self, n_classes):
        if n_classes == 'binary':
            def binary_dice_loss(pred, target):
                return 2 * torch.sum(pred * target) / (torch.sum(pred + target) + self.eps)
            return binary_dice_loss
        raise NotImplementedError


class CrossEntropyLoss(Cfg.Obj):
    with_logits = True

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


LOSSES = {
    'dice': DiceLoss,
    'cross-entropy': CrossEntropyLoss
}