from time import time

import pytorch_lightning as pl
from nn_template import CfgDict, ExperimentCfg
from nn_template.datasets import DatasetsCfg
from nn_template.training import TrainingCfg
from nn_template.hyperparameters_tuning.optuna import OptunaCfg
from nn_template.hardware import HardwareCfg
from nn_template import model
from nn_template.misc.function_tools import LogTimer

from nn_template.callbacks.log_artifacts import Export2DLabel


def run_train(cfg: CfgDict):
    experiment_cfg: ExperimentCfg = cfg['experiment']
    training_cfg: TrainingCfg = cfg['training']
    datasets_cfg: DatasetsCfg = cfg['datasets']
    optuna_cfg: OptunaCfg = cfg['optuna']
    hardware_cfg: HardwareCfg = cfg['hardware']

    # --- Setup logs ---
    wandb_log = experiment_cfg.wandb.logger

    # --- Setup seed ---
    training_cfg.configure_seed()

    # --- Setup dataset ---
    with LogTimer('Create Dataloaders', log=hardware_cfg.debug):
        _, val_data, _ = datasets_cfg.create_all_dataloaders()

    ###################
    # ---  MODEL  --- #
    # ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ #
    with LogTimer('Setup Models', log=hardware_cfg.debug):
        sample = val_data.dataset[0]
        net = cfg['model'].create(sample['x'].shape[0])
        experiment_cfg.wandb.setup_model_log(net)

    ###################
    # ---  TRAIN  --- #
    # ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ #
    callbacks = [optuna_cfg.pytorch_lightnings_callbacks()]
    trainer: pl.Trainer = cfg['training'].create_trainer(callbacks)
    trainer.fit(net)

    ################
    # --- TEST --- #
    # ‾‾‾‾‾‾‾‾‾‾‾‾ #
    cmap_av = {(0, 0): 'blue', (1, 1): 'red', (1, 0): 'cyan', (0, 1): 'pink', 'default': 'lightgray'}
    cmap_vessel = {(0, 0): 'black', (1, 1): 'white', (1, 0): 'orange', (0, 1): 'greenyellow', 'default': 'lightgray'}
    callbacks = [Export2DLabel(cmap_vessel, dataset_names=net.testset_names)]

    tester = training_cfg.create_tester(callbacks=callbacks)
    tester.test(net, ckpt_path=training_cfg.objective_best_ckpt_path)

    ###############
    # --- LOG --- #
    # ‾‾‾‾‾‾‾‾‾‾‾ #
    optuna_cfg.tell(training_cfg.objective_best_value)
    wandb_log.finalize()


