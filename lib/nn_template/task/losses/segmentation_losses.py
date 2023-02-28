import torch
from torchgeometry.losses import one_hot
from .loss_core import LossCfg, register_loss
from ...misc.clip_pad import select_pixels_by_mask

@register_loss('dice')
class DiceLoss(LossCfg):
    eps = 1e-6

    def loss(self):
        from ..task import LightningTaskCfg
        from ..segmentation2D import Segmentation2DCfg

        task: LightningTaskCfg = self.root()['task']
        if isinstance(task, Segmentation2DCfg):
            n_classes = task.n_classes
            if n_classes == 'binary':
                def binary_dice_loss(pred, target, mask):
                    pred = pred[mask]
                    target = target[mask]
                    return 2 * torch.sum(pred * target) / (torch.sum(pred + target) + self.eps)
                return binary_dice_loss
            else:
                def dice_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                    # Remove masked target
                    pred, target = select_pixels_by_mask(pred, target, mask=mask)

                    # Create the labels one hot tensor
                    target_one_hot = torch.zeros(pred.shape, device=target.device, dtype=target.dtype,)
                    target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)

                    # compute the dice score
                    intersection = torch.sum(pred * target_one_hot, (1,))
                    cardinality = torch.sum(pred + target_one_hot, (1,))

                    dice_score = intersection / (cardinality + self.eps)

                    # Average on the batch dimension
                    return 1. - 2.*torch.mean(dice_score)
                return dice_loss

        raise NotImplementedError('DiceLoss is only available for Segmentation2D task')
