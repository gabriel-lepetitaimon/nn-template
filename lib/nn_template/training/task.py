import inspect
import torch
import wandb
from torch import nn
import torch.optim as torch_opti
import pytorch_lightning as pl

from .. import Cfg
from .clip_pad import clip_pad_center


class LossesCfg(Cfg.Obj):
    type = Cfg.oneOf('cross-entropy', 'dice')
    eps = 1e-6

    def create(self, n_classes):
        match self.type:
            case 'cross-entropy':
                if n_classes == 'binary':
                    def binary_cross_entropy(pred, target):
                        return F.binary_cross_entropy_with_logits(pred, target.float())
                    return binary_cross_entropy
                else:
                    def cross_entropy(pred, target):
                        return F.cross_entropy(pred, target.float())
                    return cross_entropy
            case 'dice':
                if n_classes == 'binary':
                    def binary_dice_loss(pred, target):
                        pred = torch.sigmoid(pred)
                        return 2 * torch.sum(pred * target) / (torch.sum(pred + target) + self.eps)
                    return binary_dice_loss
        raise NotImplementedError()


class OptimizerCfg(Cfg.Obj):
    type = Cfg.strMap({
        'Adam': torch_opti.Adam,
        'AdamW': torch_opti.AdamW,
        'Adamax': torch_opti.Adamax,
        'ASGD': torch_opti.ASGD,
        'SGD': torch_opti.SGD,
    })
    lr = Cfg.float()

    def create(self, parameters):
        Optimizer = self.type
        arch_opts = list(inspect.signature(Optimizer).parameters.values())
        cfg = dict()
        attr = self.attr()
        for k, v in self.items():
            if k == 'type' or k in cfg:
                continue
            elif k in attr:
                cfg[k] = attr[k]
            elif k in arch_opts:
                cfg[k] = v
            else:
                print(f'Warning! Useless parameter "{k}" for optimizer {self["type"]}.')
        return Optimizer(parameters, **cfg)


@Cfg.register_obj("task", type='Segmentation2D')
class Segmentation2DCfg(Cfg.Obj):
    metrics = Cfg.strList('acc')
    objective = Cfg.str('acc')
    direction = Cfg.oneOf('max', 'min', default='max')

    n_classes = Cfg.str('binary')
    loss: LossesCfg = Cfg.obj(default='cross-entropy', shortcut='type')
    optimizer: OptimizerCfg = Cfg.obj(default='Adam', shortcut='type')

    @n_classes.checker
    def check_n_classes(self, value):
        if value == 'binary':
            return value
        return int(value)

    @objective.checker
    def check_objective(self, value):
        value = value.strip()
        if value not in self.metrics + ['val-'+m for m in self.metrics]:
            raise Cfg.InvalidAttr(f'Unknown metric "{value}" provided for attribute {self.name}',
                                  f"Valid metrics are {', '.join(self.metrics)}.")
        return value


# ==================================================================================
class Segmentation2D(pl.LightningModule):
    def __init__(self, cfg: Segmentation2DCfg, model: nn.Module):
        super(Segmentation2D, self).__init__()
        self.cfg = cfg
        self.model = model
        self.loss = self.cfg.loss.create(self.cfg.n_classes)

    def forward(self, *args, **kwargs):
        if self.cfg.n_classes == 'binary':
            return torch.sigmoid(self.model(*args, **kwargs))

    def compute_pred_target(self, batch):
        x, target = batch['x'], batch['y']
        pred = self.model(x, **{k: v for k, v in batch.items() if k not in ('x', 'y', 'mask')})
        if self.cfg.n_classes == 'binary' and pred.ndim == 4:
            pred = pred.squeeze(1)
        target = clip_pad_center(target, pred.shape)
        return pred, target

    def select_by_mask(self, *tensors, mask):
        mask = mask.to(torch.bool)
        clipped_mask = None
        selected = []
        for t in tensors:
            if clipped_mask is None or t.shape[-2:] != clipped_mask.shape:
                clipped_mask = clip_pad_center(mask, t.shape)
            if clipped_mask.ndim > t.ndim:
                clipped_mask.squeeze(1)
            elif clipped_mask.ndim < t.ndim:
                clipped_mask.unsqueeze(1)
            selected += [t[clipped_mask]]
        return selected

    def training_step(self, batch, batch_idx):
        pred, target = self.compute_pred_target(batch)
        mask = batch.get('mask', None)
        if mask is not None:
            pred, target = self.select_by_mask(pred, target, mask=mask)
        else:
            pred, target = pred.flatten(), target.flatten()
        loss = self.loss(pred, target)
        self.log('train-loss', loss.detach().cpu().item(),
                 on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return loss

