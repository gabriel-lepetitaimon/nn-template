import torch
from .loss_core import Loss, register_loss


@register_loss('dice')
class DiceLoss(Loss):
    eps = 1e-6

    def create(self, n_classes):
        if n_classes == 'binary':
            def binary_dice_loss(pred, target):
                return 2 * torch.sum(pred * target) / (torch.sum(pred + target) + self.eps)
            return binary_dice_loss
        raise NotImplementedError
