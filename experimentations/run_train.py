import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from orion.client import report_objective
import os
import os.path as P
from json import dump

from src.config import parse_arguments
from src.datasets import load_dataset
from src.trainer import Binary2DSegmentation, ExportValidation
from src.trainer.loggers import Logs
from steered_cnn.models import setup_model


def run_train(**opt):
    # --- Parse cfg ---
    cfg = parse_arguments(opt)
    args = cfg['script-arguments']

    # --- Set Seed --
    seed = cfg.training.get('seed', None)
    if seed == "random":
        seed = int.from_bytes(os.getrandom(32), 'little', signed=False)
    elif isinstance(seed, (tuple, list)):
        seed = seed[cfg.trial.ID % len(seed)]
    if isinstance(seed, int):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cfg.training['seed'] = seed
    elif seed is not None:
        print(f"Seed can't be interpreted as int and will be ignored.")

    # --- Setup logs ---
    logs = Logs()
    logs.setup_log(cfg)
    tmp_path = logs.tmp_path

    # --- Setup dataset ---
    trainD, validD, testD = load_dataset(cfg)

    ###################
    # ---  MODEL  --- #
    # ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ #
    sample = validD.dataset[0]
    model = setup_model(cfg['model'], n_in=sample['x'].shape[0], 
                        n_out=1 if sample['y'].ndim==2 else sample['y'].shape[0])

    model_inputs = {'x': '@x'}
    steering_field = cfg.get('model.steered.steering', None)
    if isinstance(steering_field, str) and steering_field != 'attention':
        if 'datasets' in cfg:
            model_inputs['alpha'] = '@'+steering_field
        else:
            model_inputs['alpha'] = '@alpha'
    hyper_params = cfg['hyper-parameters']

    net = Binary2DSegmentation(model=model, loss=hyper_params['loss'],
                               pos_weighted_loss=hyper_params['pos-weighted-loss'],
                               soft_label=hyper_params['smooth-label'],
                               earlystop_cfg=cfg['training']['early-stopping'],
                               optimizer=hyper_params['optimizer'],
                               lr=hyper_params['lr'] / hyper_params['accumulate-gradient-batch'],
                               p_dropout=hyper_params['drop-out'],
                               model_inputs=model_inputs)
    logs.log_miscs({'model': {
        'params': sum(p.numel() for p in net.parameters())
    }})

    ###################
    # ---  TRAIN  --- #
    # ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ #
    val_n_epoch = cfg.training['val-every-n-epoch']
    max_epoch = cfg.training['max-epoch']

    # Define r_code, a return code sended back to train_single.py.
    # (A run is considered successful if it returns 10 <= r <= 20. Otherwise orion is interrupted.)
    r_code = 10

    trainer_kwargs = {}
    if cfg.training['half-precision']:
        trainer_kwargs['amp_level'] = 'O2'
        trainer_kwargs['precision'] = 16

    callbacks = []
    if cfg.training['early-stopping']['monitor'].lower() != 'none':
        if cfg.training['early-stopping']['monitor'].lower() == 'auto':
            cfg.training['early-stopping']['monitor'] = cfg.training['optimize']
        earlystop = EarlyStopping(verbose=False, strict=False, **cfg.training['early-stopping'])
        callbacks += [earlystop]
    else:
        earlystop = None
    callbacks += [LearningRateMonitor(logging_interval='epoch')]

    checkpointed_metrics = ['val-acc', 'val-roc', 'val-iou']
    modelCheckpoints = {}
    for metric in checkpointed_metrics:
        checkpoint = ModelCheckpoint(dirpath=tmp_path + '/', filename='best-'+metric+'-{epoch}', monitor=metric, mode='max')
        modelCheckpoints[metric] = checkpoint
        callbacks.append(checkpoint)
        
    trainer = pl.Trainer(gpus=args.gpus, callbacks=callbacks, logger=logs.loggers,
                         max_epochs=int(np.ceil(max_epoch / val_n_epoch) * val_n_epoch),
                         check_val_every_n_epoch=val_n_epoch,
                         accumulate_grad_batches=cfg['hyper-parameters']['accumulate-gradient-batch'],
                         progress_bar_refresh_rate=1 if args.debug else 0,
                         **trainer_kwargs)

    try:
        trainer.fit(net, trainD, validD)
    except KeyboardInterrupt:
        r_code = 1  # Interrupt Orion

    logs.log_metric('last-epoch', earlystop.stopped_epoch if earlystop is not None else max_epoch)

    ################
    # --- TEST --- #
    # ‾‾‾‾‾‾‾‾‾‾‾‾ #
    reported_metric = cfg.training['optimize']
    best_ckpt = None
    if reported_metric not in modelCheckpoints:
        print('\n!!!!!!!!!!!!!!!!!!!!')
        print(f'>> Invalid optimized metric {reported_metric}, optimizing {checkpointed_metrics[0]} instead.')
        print('')
        reported_metric = checkpointed_metrics[0]
    for metric_name, checkpoint in modelCheckpoints.items():
        metric_value = float(checkpoint.best_model_score.cpu().numpy())
        logs.log_metrics({'best-' + metric_name: metric_value,
                          f'best-{metric_name}-epoch': float(checkpoint.best_model_path[:-5].rsplit('-', 1)[1][6:])})
        if metric_name == reported_metric:
            best_ckpt = checkpoint
            reported_value = -metric_value
    
    if 'av' in cfg.training['dataset-file']:
        cmap = {(0, 0): 'blue', (1, 1): 'red', (1, 0): 'cyan', (0, 1): 'pink', 'default': 'lightgray'}
    else:
        cmap = {(0, 0): 'black', (1, 1): 'white', (1, 0): 'orange', (0, 1): 'greenyellow', 'default': 'lightgray'}

    net.testset_names, testD = list(zip(*testD.items()))
    tester = pl.Trainer(gpus=args.gpus, logger=logs.loggers,
                        callbacks=[ExportValidation(cmap, path=tmp_path + '/samples', dataset_names=net.testset_names)],
                        progress_bar_refresh_rate=1 if args.debug else 0,)
    tester.test(net, testD, ckpt_path=best_ckpt.best_model_path)

    ###############
    # --- LOG --- #
    # ‾‾‾‾‾‾‾‾‾‾‾ #
    report_objective(reported_value)
    logs.save_cleanup()

    # Store data in a json file to send info back to train_single.py script.
    with open(P.join(cfg['script-arguments']['tmp-dir'], f'result.json'), 'w') as f:
        json = {'r_code': r_code}
        dump(json, f)
        print("WRITING JSON AT: ", P.join(cfg['script-arguments']['tmp-dir'], f'result.json'))


def leg_setup_model(model_cfg, old=False):
    from steered_cnn.models import HemelingNet, SteeredHemelingNet, OldHemelingNet
    if model_cfg['steered']:
        model = SteeredHemelingNet(6, nfeatures_base=model_cfg['nfeatures-base'],
                                   padding=model_cfg['padding'],
                                   depth=model_cfg['depth'],
                                   batchnorm=model_cfg['batchnorm'],
                                   upsample=model_cfg['upsample'],
                                   attention=model_cfg['steered'] == 'attention')
    else:
        if old:
            model = OldHemelingNet(6, nfeatures_base=model_cfg['nfeatures-base'],
                                padding=model_cfg['padding'],
                                half_kernel_height=model_cfg['half-kernel-height'])
        else:
            model = HemelingNet(6, nfeatures_base=model_cfg['nfeatures-base'],
                                padding=model_cfg['padding'],
                                depth=model_cfg['depth'],
                                batchnorm=model_cfg['batchnorm'],
                                half_kernel_height=model_cfg['half-kernel-height'])
    return model


if __name__ == '__main__':
    run_train()
