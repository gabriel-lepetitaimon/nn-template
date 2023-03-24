from nntemplate import CfgDict
from nntemplate.experiment import ExperimentCfg
from nntemplate.datasets import DatasetsCfg
from nntemplate.training import TrainingCfg
from nntemplate.hyperparameters_tuning.optuna import OptunaCfg
from nntemplate.hardware import HardwareCfg
from nntemplate.task.segmentation2D import Segmentation2DCfg
from nntemplate.model import SMPModelCfg
from nntemplate.torch_utils.function_tools import LogTimer

from nntemplate.callbacks.log_artifacts import Export2DLabel


def run_train(cfg: CfgDict):
    """
    Perform a single training run with the given configuration.
    Args:
        cfg (CfgDict): Configuration dictionary.

    Returns:

    """
    experiment_cfg: ExperimentCfg = cfg['experiment']
    training_cfg: TrainingCfg = cfg['training']
    task_cfg: Segmentation2DCfg = cfg['task']
    datasets_cfg: DatasetsCfg = cfg['datasets']
    optuna_cfg: OptunaCfg = cfg['optuna']
    model_cfg: SMPModelCfg = cfg['model']
    hardware_cfg: HardwareCfg = cfg['hardware']

    cmap_av = {(0, 0): 'blue', (1, 1): 'red', (1, 0): 'cyan', (0, 1): 'pink', 'default': 'lightgray'}
    cmap_vessel = {(0, 0): '#edf6f9', (1, 1): '#83c5be', (1, 0): '#e29578', (0, 1): '#006d77', 'default': 'lightgray'}

    # --- Setup logs ---
    with experiment_cfg.wandb.log_run() as wandb_log:

        # --- Setup seed ---
        training_cfg.configure_seed()

        # --- Setup dataset ---
        with LogTimer('Create Dataloaders', log=hardware_cfg.debug):
            train_data, val_data = datasets_cfg.create_train_val_dataloaders()

        ###################
        # ---  MODEL  --- #
        # ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ #
        with LogTimer('Setup Models', log=hardware_cfg.debug):
            sample = val_data.dataset[0]
            model = model_cfg.create(sample['x'].shape[0])

            callbacks = experiment_cfg.wandb.pl_callbacks()

        print('\t==== MODEL SPECS ===')
        print(model)

        ###################
        # ---  TRAIN  --- #
        # ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ #
        with LogTimer('Setup Trainer', log=hardware_cfg.debug):
            callbacks += optuna_cfg.pl_callbacks()
            callbacks += [Export2DLabel(cmap_vessel, every_n_epoch=10)]
            trainer = training_cfg.create_trainer(callbacks, logger=wandb_log)
            net = task_cfg.create_lightning_task(model)

        trainer.fit(net, train_data, val_data)

        del train_data
        del val_data

        ################
        # --- TEST --- #
        # ‾‾‾‾‾‾‾‾‾‾‾‾ #
        with LogTimer('Setup Tester', log=hardware_cfg.debug):
            callbacks = [Export2DLabel(cmap_vessel, dataset_names=net.test_dataloaders_names)]

            tester = training_cfg.create_tester(callbacks=callbacks, logger=wandb_log)
            test_data = datasets_cfg.create_test_dataloaders()

        tester.test(net, test_data, ckpt_path=training_cfg.objective_best_ckpt_path)

    return training_cfg.objective_best_value

