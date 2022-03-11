from functools import partial

# import mlflow
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import functional as metricsF
import torchmetrics

from steered_cnn.utils import clip_pad_center
from ..config import default_config


class Binary2DSegmentation(pl.LightningModule):
    def __init__(self, model, model_inputs=None,
                 loss='binaryCE', pos_weighted_loss=False,
                 optimizer=None, earlystop_cfg=None,
                 lr=1e-3, p_dropout=0, soft_label=0):
        super().__init__()
        self.model = model
        self.model_inputs = {'x': 'x'} if model_inputs is None else model_inputs

        self.val_accuracy = torchmetrics.Accuracy(compute_on_step=False)
        self.lr = lr
        self.p_dropout = p_dropout
        self.soft_label = soft_label
        
        if isinstance(loss, dict):
            loss_kwargs = loss
            loss = loss['type']
            del loss_kwargs['type']
        else:
            loss_kwargs = {}

        self.pos_weighted_loss = pos_weighted_loss
        if pos_weighted_loss:
            if loss == 'binaryCE':
                self._loss = lambda y_hat, y, weight: F.binary_cross_entropy_with_logits(y_hat, y.float(),
                                                                                         pos_weight=weight)
            else:
                raise ValueError(f'Invalid weighted loss function: "{loss}". (Only "binaryCE" is supported.)')
        else:
            if loss == 'dice':
                from .losses import binary_dice_loss
                self._loss = lambda y_hat, y: binary_dice_loss(torch.sigmoid(y_hat), y)
            elif loss == 'focalLoss':
                from .losses import focal_loss
                _loss = partial(focal_loss, gamma=loss_kwargs.get('gamma',2))
                self._loss = lambda y_hat, y: _loss(torch.sigmoid(y_hat), y)
            elif loss == 'binaryCE':
                self._loss = lambda y_hat, y: F.binary_cross_entropy_with_logits(y_hat, y.float())
            else:
                raise ValueError(f'Unkown loss function: "{loss}". \n'
                                 f'Should be one of "dice", "focalLoss", "binaryCE".')

        if optimizer is None:
            optimizer = {'type': 'Adam'}
        self.optimizer = optimizer
        if earlystop_cfg is None:
            earlystop_cfg= default_config()['training']['early-stopping']
        self.earlystop_cfg = earlystop_cfg

        self.testset_names = None
        
    def loss_f(self, pred, target, weight=None):
        if self.soft_label:
            target = target.float()
            target *= 1-2*self.soft_label
            target += self.soft_label
        if self.pos_weighted_loss:
            return self._loss(pred, target, weight)
        else:
            return self._loss(pred, target)

    def compute_y_yhat(self, batch):
        y = (batch['y'] != 0).int()
        model_inputs = {model_arg: batch[field[1:]] if isinstance(field, str) and field.startswith('@') else field
                        for model_arg, field in self.model_inputs.items()}
        y_hat = self.model(**model_inputs).squeeze(1)
        y = clip_pad_center(y, y_hat.shape)
        return y, y_hat

    def training_step(self, batch, batch_idx):
        y, y_hat = self.compute_y_yhat(batch)

        mask = None
        if 'mask' in batch:
            mask = clip_pad_center(batch['mask'], y_hat.shape)
            thr_mask = mask!=0
            y_hat = y_hat[thr_mask].flatten()
            y = y[thr_mask].flatten()
            if self.pos_weighted_loss:
                mask = mask[thr_mask].flatten()
        if self.pos_weighted_loss:
            loss = self.loss_f(y_hat, y, mask)
        else:
            loss = self.loss_f(y_hat, y)
        loss_value = loss.detach().cpu().item()
        self.log('train-loss', loss_value, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return loss

    def _validate(self, batch):
        y, y_hat = self.compute_y_yhat(batch)
        
        y_sig = torch.sigmoid(y_hat)
        y_pred = y_sig > .5
        
        if 'mask' in batch:
            mask = clip_pad_center(batch['mask'], y_hat.shape)
            y_hat = y_hat[mask != 0]
            y_sig = y_sig[mask != 0]
            y = y[mask != 0]
            
        y = y.flatten()
        y_hat = y_hat.flatten()
        y_sig = y_sig.flatten()

        return {
            'loss': self.loss_f(y_hat, y),
            'y_pred': y_pred,
            'y_hat': y_hat,
            'y': y,
            'y_sig': y_sig,
            'metrics': Binary2DSegmentation.metrics(y_sig, y)
        }

    @staticmethod
    def metrics(y_sig, y):
        y_pred = y_sig > 0.5
        return {
            'acc': metricsF.accuracy(y_pred, y),
            'roc': metricsF.auroc(y_sig, y),
            'iou': metricsF.iou(y_pred, y),
        }

    def log_metrics(self, metrics, prefix='', discard_dataloaderidx=False):
        if prefix and not prefix.endswith('-'):
            prefix += '-'
        for k, v in metrics.items():
            if discard_dataloaderidx:
                idx = self._current_dataloader_idx
                self._current_dataloader_idx = None
            self.log(prefix + k, v.cpu().item())
            if discard_dataloaderidx:
                self._current_dataloader_idx = idx

    def validation_step(self, batch, batch_idx):
        result = self._validate(batch)
        metrics = result['metrics']
        # metrics['acc'] = self.val_accuracy(result['y_sig'] > 0.5, result['y'])
        self.log_metrics(metrics, 'val', discard_dataloaderidx=True)
        return result['y_pred']

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        result = self._validate(batch)
        metrics = result['metrics']
        prefix = 'test'
        if self.testset_names:
            prefix = self.testset_names[dataloader_idx]
        self.log_metrics(metrics, prefix, discard_dataloaderidx=True)
        return result['y_pred']

    def configure_optimizers(self):
        opt = self.optimizer
        if opt['type'].lower() in ('adam', 'adamax', 'adamw'):
            Adam = {'adam': torch.optim.Adam,
                    'adamax': torch.optim.Adamax,
                    'adamw': torch.optim.AdamW}[opt['type'].lower()]
            kwargs = {k: v for k, v in opt.items() if k in ('weight_decay', 'amsgrad', 'eps')}
            optimizer = Adam(self.parameters(), lr=self.lr, betas=(opt.get('beta', .9), opt.get('beta_sqr', .999)),
                             **kwargs)
        elif opt['type'].lower() == 'asgd':
            kwargs = {k: v for k, v in opt.items() if k in ('lambd', 'alpha', 't0', 'weight_decay')}
            optimizer = torch.optim.ASGD(self.parameters(), lr=self.lr, **kwargs)
        elif opt['type'].lower() == 'sgd':
            kwargs = {k: v for k, v in opt.items() if k in ('momentum', 'dampening', 'nesterov', 'weight_decay')}
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, **kwargs)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if opt['lr-decay-factor']:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.earlystop_cfg['mode'],
                                                                   factor=opt['lr-decay-factor'],
                                                                   patience=self.earlystop_cfg['patience']/2,
                                                                   threshold=self.earlystop_cfg['min_delta'],
                                                                   min_lr=self.lr*opt['lr-decay-factor']**5)
            return {'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'frequency': 1, 'interval': 'epoch',
                        'monitor': self.earlystop_cfg['monitor'],

                    }}
        else:
            return optimizer

    def forward(self, *args, **kwargs):
        return torch.sigmoid(self.model(*args, **kwargs))

    def test(self, datasets):
        if isinstance(datasets, dict):
            self.testset_names, datasets = list(zip(*datasets.items()))
        trainer = pl.Trainer(gpus=[0])
        return trainer.test(self, test_dataloaders=datasets)

    @property
    def p_dropout(self):
        return self.model.p_dropout

    @p_dropout.setter
    def p_dropout(self, p):
        self.model.p_dropout = p
