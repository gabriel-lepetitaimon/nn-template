import argparse
import traceback
from nn_template import Cfg
from nn_template.hyperparameters_tuning.optuna import OptunaCfg

from .run_train import run_train


def main():
    parser = argparse.ArgumentParser(prog='TrainSingleNN',
                                     description='Parse a single configuration file and perform training runs until '
                                                 'completion of the hyperparameter search.')
    parser.add_argument('configuration_file')
    parser.add_argument('-g', '--gpus')
    parser.add_argument('-D', '--debug', action='store_true')
    args = parser.parse_args()

    override_cfg = {
        'hardware': {
            'gpus': args.gpus,
            'debug': args.debug
        }
    }

    try:
        exhaust_runs(args.configuration_file, override_cfg)
    except KeyboardInterrupt:
        print('\n'*2)
        print("====== RUN WAS INTERRUPTED =======")
        print('\n' * 2, '-'*50, '\n')
        exit(100)
    except Exception as e:
        print('\n'*2)
        print("========== RUN CRASHED !! ========")
        traceback.print_exc()
        print('\n' * 2, '-'*50, '\n')
        exit(1)


def exhaust_runs(cfg_filepath: str, override_cfg=None):
    for cfg in Cfg.Parser(cfg_filepath, override=override_cfg).get_configs():
        optuna_cfg: OptunaCfg = cfg['optuna']
        while not optuna_cfg.hyper_parameter_search_complete():
            with optuna_cfg.trial_ctx() as trial_cfg:    # Suggest hyperparameter
                run_train(trial_cfg)


if __name__ == '__main__':
    main()
