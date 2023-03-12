# Neural Network Template

[![PyPI version](https://badge.fury.io/py/nntemplate.svg)](https://badge.fury.io/py/nntemplate)

Neural Network Template is a Python library for training neural networks on image classification and segmentation tasks.

## Installation
nntemplate requires Python 3.10 or higher. To install the latest version from PyPI:

```
pip install -U nntemplate
```

## A simple example
```python
import nntemplate as nnt
from nntemplate import Cfg
from nntemplate.hyperparameters_tuning.optuna import OptunaCfg
from nntemplate.experiment import ExperimentCfg
from nntemplate.datasets import DatasetsCfg
from nntemplate.task import Segmentation2DCfg
from nntemplate.training import TrainingCfg

cfg = nnt.parse('path/to/config.yaml').get_config()
optuna_cfg: OptunaCfg = cfg['optuna']
experiment_cfg: ExperimentCfg = cfg['experiment']
task_cfg: Segmentation2DCfg = cfg['task']
train_cfg: TrainingCfg = cfg['training']
datasets_cfg: DatasetsCfg = cfg['datasets']

with experiment_cfg.wandb.log_run():
    # Create the dataloaders and the model
    train, val, test = datasets_cfg.create_all_dataloaders()
    model = task_cfg.create_lightning_task()
    
    # Train the model
    trainer = train_cfg.create_trainer()
    trainer.fit(model, train, val)
    
    # Evaluate the model on the test set
    tester = train_cfg.create_tester()
    tester.test(model, test, ckpt_path=train_cfg.objective_best_ckpt_path)
```




