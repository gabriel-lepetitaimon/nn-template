import torch

def binary_dice_loss(pred, target):
    eps = 1e-6
    dice = 2*torch.sum(pred*target)/(torch.sum(pred+target)+eps)
    return 1-dice


def focal_loss(pred, target, gamma=2):
    pt = pred*target + (1-target)*(1-pred)
    return - torch.mean((1-pt)**gamma * torch.log(pt))
