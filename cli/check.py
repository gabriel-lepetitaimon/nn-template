import argparse

from nntemplate import Cfg
from run_train import *


def main():
    parser = argparse.ArgumentParser(prog='CheckCfg',
                                     description='Parse a single configuration file and perform quick checks '
                                                 'on datasets and on the model.')

    parser.add_argument('configuration_file')
    parser.add_argument('-g', '--gpus')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    override_cfg = {}
    if args.gpus:
        override_cfg['hardware.gpus'] = args.gpus
    override_cfg['hardware.debug'] = True
    check(args.configuration_file, override_cfg, args.verbose)


def check(cfg_filepath: str, override_cfg=None, verbose=False):
    parser = Cfg.Parser(cfg_filepath, override=override_cfg).parse()

    for cfg_i in range(len(parser)):
        print(f'CHECKING CONFIG: {cfg_filepath}' + (f' - version {cfg_i}' if len(parser) > 1 else ''))
        with LogTimer('Parsing configuration', log=verbose):
            cfg = parser.get_config(cfg_i)

        dataset_cfg: DatasetsCfg = cfg['datasets']
        model_cfg = cfg['model']

        with LogTimer('Building Train & Val Dataloader', log=verbose):
            train_data, val_data = dataset_cfg.create_train_val_dataloaders()

        with LogTimer('Generate one batch of training data', log=verbose):
            if verbose:
                print('\tTrain data length:', len(train_data.dataset))
            for d in train_data:
                if verbose:
                    print('\tTrain data shape | ' + ', '.join(f'{k}: {tuple(v.shape)}' for k, v in d.items()))
                n_in = d['x'].shape[1]
                break
        del train_data

        with LogTimer('Generate one batch of validation data', log=verbose):
            if verbose:
                print('\tVal data length:', len(val_data.dataset))
            for d in val_data:
                if verbose:
                    print('\tVal data shape | ' + ', '.join(f'{k}: {tuple(v.shape)}' for k, v in d.items()))
                assert n_in == d['x'].shape[1], 'Number of input features must be the same for train and val data.'
                break
        del val_data

        with LogTimer('Building Test Dataloaders', log=verbose):
            test_datas = dataset_cfg.create_test_dataloaders()

        for test_name, test_data in zip(dataset_cfg.test_datasets_names, test_datas):
            with LogTimer(f'Generate one batch from: {test_name}', log=verbose):
                if verbose:
                    print(f'\t{test_name} length:  {len(test_data.dataset)}')
                for d in test_data:
                    sample = d
                    if verbose:
                        print(f'\t{test_name} data shape | ' + ', '.join(f'{k}: {tuple(v.shape)}' for k, v in d.items()))
                    assert n_in == d['x'].shape[
                        1], f'Number of input features must be the same for train and {test_name} data.'
                    break
        del test_datas

        with LogTimer('Check Model', log=verbose):
            model = model_cfg.create()
            if verbose:
                try:
                    from torchinfo import summary
                    summary(model, sample['x'].shape, device='cuda')
                except ImportError:
                    print(model)
        del model

        print(f'Config is Valid!  ({cfg_filepath}' + (f' - version {cfg_i}' if len(parser) > 1 else '') + ')')


if __name__ == '__main__':
    main()
