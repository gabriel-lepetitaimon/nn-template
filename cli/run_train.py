from torchinfo import summary

from nntemplate import CfgDict
from nntemplate.experiment import ExperimentCfg
from nntemplate.datasets import DatasetsCfg
from nntemplate.training import TrainingCfg
from nntemplate.hyperparameters_tuning.optuna import OptunaCfg
from nntemplate.hardware import HardwareCfg
from nntemplate.task.segmentation2D import Segmentation2DCfg
from nntemplate.model import SMPModelCfg
from nntemplate.utils.function_tools import LogTimer


def run_train(cfg: CfgDict, train_callbacks=(), test_callbacks=()):
    """
    Perform a single training run with the given configuration.
    Args:
        cfg (CfgDict): Configuration dictionary.
        train_callbacks (list): List of callbacks to use during training.
        test_callbacks (list): List of callbacks to use during testing.

    Returns:
        The best value of the optimized metric. (cf. `cfg.training.objective`)
    """
    experiment_cfg: ExperimentCfg = cfg['experiment']
    training_cfg: TrainingCfg = cfg['training']
    task_cfg: Segmentation2DCfg = cfg['task']
    datasets_cfg: DatasetsCfg = cfg['datasets']
    optuna_cfg: OptunaCfg = cfg['optuna']
    model_cfg: SMPModelCfg = cfg['model']
    hardware_cfg: HardwareCfg = cfg['hardware']

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
        callbacks = list(train_callbacks)

        with LogTimer('Setup Models', log=hardware_cfg.debug):
            sample = val_data.dataset[0]
            model = model_cfg.create(sample['x'].shape[0])

            callbacks += experiment_cfg.wandb.pl_callbacks()

        print('\t==== MODEL SPECS ===')
        summary(model, sample['x'].shape, device='cuda')

        ###################
        # ---  TRAIN  --- #
        # ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ #
        with LogTimer('Setup Trainer', log=hardware_cfg.debug):
            callbacks += optuna_cfg.pl_callbacks()
            callbacks += []
            net = task_cfg.create_task(model)
            trainer = training_cfg.create_trainer(callbacks, logger=wandb_log)

        trainer.fit(net, train_data, val_data)

        del train_data
        del val_data

        ################
        # --- TEST --- #
        # ‾‾‾‾‾‾‾‾‾‾‾‾ #
        with LogTimer('Setup Tester', log=hardware_cfg.debug):
            tester = training_cfg.create_tester(callbacks=test_callbacks, logger=wandb_log)
            test_data = datasets_cfg.create_test_dataloaders()

        tester.test(net, test_data, ckpt_path=training_cfg.objective_best_ckpt_path)

    return training_cfg.objective_best_value

