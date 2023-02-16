import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, StochasticWeightAveraging
from orion.client import report_objective
import os.path as P
from json import dump

from nn_template import CfgDict, ExperimentCfg
from nn_template.datasets import DatasetsCfg
from nn_template.training import TrainingCfg
from nn_template.hyperparameters_tuning.optuna import OptunaCfg
from nn_template import model

from src.trainer import Binary2DSegmentation, ExportValidation


def run_train(cfg: CfgDict):
    experiment_cfg: ExperimentCfg = cfg['experiment']
    training_cfg: TrainingCfg = cfg['training']
    datasets_cfg: DatasetsCfg = cfg['datasets']
    optuna_cfg: OptunaCfg = cfg['optuna']

    # --- Setup logs ---
    wandb_log = experiment_cfg.wandb.logger

    # --- Setup seed ---
    training_cfg.configure_seed()

    # --- Setup dataset ---
    train_data, val_data, test_datasets = datasets_cfg.create_all_dataloaders()

    ###################
    # ---  MODEL  --- #
    # ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ #
    sample = val_data.dataset[0]
    net = cfg['model'].create(sample['x'].shape[0])
    experiment_cfg.wandb.setup_model_log(net)

    ###################
    # ---  TRAIN  --- #
    # ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ #
    callbacks = [optuna_cfg.pytorch_lightnings_callbacks()]
    trainer: pl.Trainer = cfg['training'].create_trainer(callbacks, train_data, val_data)
    trainer.fit(net)

    ################
    # --- TEST --- #
    # ‾‾‾‾‾‾‾‾‾‾‾‾ #
    
    if 'av' in ['dataset-file']:
        cmap = {(0, 0): 'blue', (1, 1): 'red', (1, 0): 'cyan', (0, 1): 'pink', 'default': 'lightgray'}
    else:
        cmap = {(0, 0): 'black', (1, 1): 'white', (1, 0): 'orange', (0, 1): 'greenyellow', 'default': 'lightgray'}

    callbacks = [ExportValidation(cmap, dataset_names=net.testset_names)]

    tester = training_cfg.create_tester(callbacks=callbacks)
    tester.test(net, ckpt_path=training_cfg.objective_best_ckpt_path)

    ###############
    # --- LOG --- #
    # ‾‾‾‾‾‾‾‾‾‾‾ #
    optuna_cfg.tell(training_cfg.objective_best_value)
    wandb_log.finalize()


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
