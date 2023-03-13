import argparse

from nntemplate import Cfg
from nntemplate.misc.function_tools import LogTimer
from nntemplate.datasets import DatasetsCfg


def check(cfg_filepath: str, override_cfg=None):
    parser = Cfg.Parser(cfg_filepath, override=override_cfg).parse()

    for cfg_i in range(len(parser)):
        print(f'CHECKING CONFIG: {cfg_filepath}' + (f' - version {cfg_i}' if len(parser) > 1 else ''))
        with LogTimer('Parsing configuration', log=True):
            cfg = parser.get_config(cfg_i)

        dataset_cfg: DatasetsCfg = cfg['datasets']
        model_cfg = cfg['model']

        with LogTimer('Building Train & Val Dataloader', log=True):
            train_data, val_data = dataset_cfg.create_train_val_dataloaders()

        with LogTimer('Generate one batch of training data', log=True):
            print('\tTrain data length:', len(train_data.dataset))
            for d in train_data:
                print('\tTrain data shape | '+', '.join(f'{k}: {tuple(v.shape)}' for k, v in d.items()))
                n_in = d['x'].shape[1]
                break
        del train_data

        with LogTimer('Generate one batch of validation data', log=True):
            print('\tVal data length:', len(val_data.dataset))
            for d in val_data:
                print('\tVal data shape | ' + ', '.join(f'{k}: {tuple(v.shape)}' for k, v in d.items()))
                assert n_in == d['x'].shape[1], 'Number of input features must be the same for train and val data.'
                break
        del val_data

        with LogTimer('Building Test Dataloaders', log=True):
            test_datas = dataset_cfg.create_test_dataloaders()

        for test_name, test_data in zip(dataset_cfg.test_dataloaders_names, test_datas):
            with LogTimer(f'Generate one batch from: {test_name}', log=True):
                print(f'\t{test_name} length:  {len(test_data.dataset)}')
                for d in test_data:
                    sample = d
                    print(f'\t{test_name} data shape | ' + ', '.join(f'{k}: {tuple(v.shape)}' for k, v in d.items()))
                    assert n_in == d['x'].shape[1], f'Number of input features must be the same for train and {test_name} data.'
                    break
        del test_datas

        with LogTimer('Check Model', log=True):
            model = model_cfg.create(sample['x'].shape[0])
        del model