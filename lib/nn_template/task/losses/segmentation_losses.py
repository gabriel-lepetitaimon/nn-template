import torch
from torchgeometry.losses import DiceLoss as Dice
from .loss_core import Loss, register_loss


@register_loss('dice')
class DiceLoss(Loss):
    eps = 1e-6

    def create(self):
        loss = Dice()
        loss.eps = self.eps
        return loss
