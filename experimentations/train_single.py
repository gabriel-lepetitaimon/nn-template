import argparse
from nntemplate import Cfg
from nntemplate.hyperparameters_tuning.optuna import OptunaCfg
from run_train import run_train


def main():
    parser = argparse.ArgumentParser(prog='TrainSingleNN',
                                     description='Parse a single configuration file and perform training runs until '
                                                 'completion of the hyperparameter search.')
    parser.add_argument('configuration_file')
    parser.add_argument('-g', '--gpus')
    parser.add_argument('-D', '--debug', action='store_true')
    args = parser.parse_args()

    override_cfg = {}
    if args.gpus:
        override_cfg['hardware.gpus'] = args.gpus
    if args.debug:
        override_cfg['hardware.debug'] = True

    exhaust_runs(args.configuration_file, override_cfg)


def exhaust_runs(cfg_filepath: str, override_cfg=None):
    """
    Parse the configuration file and run the training until completion of the hyperparameter search for every version
    of the file.

    Args:
        cfg_filepath: Path to the configuration file.
        override_cfg: Dictionary of configuration attributes to override.
    """
    parser = Cfg.Parser(cfg_filepath, override=override_cfg).parse()
    for cfg in parser.get_configs():

        if 'optuna' in cfg:
            optuna_cfg: OptunaCfg = cfg['optuna']
            optuna_cfg.optimize(run_train, cfg)
        else:
            run_train(cfg)


if __name__ == '__main__':
    main()
