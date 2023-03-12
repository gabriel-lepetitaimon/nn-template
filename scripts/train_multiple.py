import os
import os.path as P
import argparse
from nntemplate import Cfg
from nntemplate.hyperparameters_tuning.optuna import OptunaCfg

def main():
    # --- Parser ---
    parser = argparse.ArgumentParser(prog='TrainAllNN',
                                     description='Parse all configuration files in a directory and perform training runs.')
    parser.add_argument('cfg_path', help='Specify in which directory should the config files be read')
    parser.add_argument('--gpus', help='Specify which gpu should be used')
    parser.add_argument('--debug', action='store_true',
                        help='Enforce debug mode (max 2 epochs, no hyperparameter search, '
                             'all log are redirected to the "Debug&Test" project).')
    args = parser.parse_args()

    override_cfg = {}
    if args.gpus:
        override_cfg['hardware.gpus'] = args.gpus
    if args.debug:
        override_cfg['hardware.debug'] = True

    train_multiple(args.cfg_path, override_cfg)


def train_multiple(path=None, override_cfg=None):
    if path is None:
        path = P.abspath(P.join(P.dirname(__file__), 'EXP/'))

    for cfg in exhaust_cfg_in_path(path):

        for cfg in cfgs:
            opt = dict(config=os.path.join(path, cfg), debug=debug, gpus=gpus)
            cfg = parse_arguments(opt)
            run_experiment(cfg)


def exhaust_cfg_in_path(path: str, override_cfg=None):
    while True:
        cfg_filenames = sorted(_ for _ in os.listdir(path) if _.endswith('.yaml') if _[0] not in "!_")
        for cfg_filename in cfg_filenames:
            parser = Cfg.Parser(cfg_filename, override=override_cfg).parse()
            for cfg in parser.get_configs():

                if 'optuna' in cfg:
                    optuna_cfg: OptunaCfg = cfg['optuna']
                    optuna_cfg.optimize(run_train, cfg)
                else:
                    yield cfg


if __name__ == '__main__':
    main()
