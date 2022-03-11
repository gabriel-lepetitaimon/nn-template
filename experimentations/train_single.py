import os
import os.path as P
from typing import Dict
from json import load
import argparse
import orion.client
import orion.storage
from orion.client import get_experiment
from orion.core.utils.exceptions import NoConfigurationError
from tempfile import TemporaryDirectory

from src.config import parse_arguments, set_env_var


def main():
    # --- Parser ---
    cfg = parse_arguments()
    run_experiment(cfg)


def run_experiment(cfg):
    script_args = cfg['script-arguments']
    DEBUG = script_args.debug
    cfg_path = script_args.config

    # --- Parse Config ---
    exp_cfg = cfg.experiment
    orion_exp_name = f"{exp_cfg.name}-{exp_cfg['sub-experiment']}-{exp_cfg['sub-experiment-id']:03}"

    if not DEBUG:
        orion.storage.base.setup_storage(cfg.orion.storage.to_dict())

    ended = False
    while not ended:
        cfg['trial'] = dict(ID=0, name=orion_exp_name, version=0, cfg_path=cfg_path)
        # --- Fetch Orion Infos ---
        if not DEBUG:
            try:
                orion_exp = get_experiment(orion_exp_name)
                trial_id = cfg['trial']['ID'] = len(orion_exp.fetch_trials())
            except NoConfigurationError:
                pass
            else:
                if orion_exp.is_done:
                    print(f'!> Orion Experiment is done (trail id={trial_id}/{orion_exp.max_trials}). \n'
                          f'!> Exiting orion experiment "{orion_exp_name}".')
                    return True
                elif orion_exp.is_broken:
                    print(f'!> Orion Experiment is broken (trail id={trial_id}/{orion_exp.max_trials}). \n'
                          f'!> Exiting orion experiment "{orion_exp_name}".')
                    return False
                else:
                    cfg['trial']['ID'] = trial_id
                    cfg['trial']['version'] = orion_exp.version

        print('')
        print(f' === Running {orion_exp_name} ({cfg_path}): trial {cfg["trial"]["ID"]} ===')
        r = run_orion(cfg)


        if 10 <= r.get('r_code', -10) <= 20:
            print('')
            print('-'*30)
            print('')
            if DEBUG:
                print(f'!> Debug trial run smoothly! Exiting....\n')
                return True
            continue
        else:
            print(f'!> Trial {cfg["trial"]["ID"]} exited with r_code={r.get("r_code", -10)}.')
            if 'error' in r:
                print(f'!> ERROR: ' + r.get('error'))
            elif 'r_code' not in r:
                print(f'!> MISSING r_code in:\n{repr(r)}'.replace("\n", "\n\t>"))
            print(f'!> Exiting orion experiment "{orion_exp_name}".')
            return DEBUG


def run_orion(cfg: Dict):
    script_args = cfg['script-arguments']
    orion_exp_name = cfg['trial']['name']
    cfg_path = script_args['config']
    DEBUG = script_args['debug']

    # --- Prepare tmp folder ---
    tmp_path = script_args['tmp-dir']
    if not P.exists(tmp_path):
        os.makedirs(tmp_path)
    with TemporaryDirectory(dir=tmp_path, prefix=f"{orion_exp_name}-{cfg['trial']['ID']}") as tmp_dir:
        cfg['script-arguments']['tmp-dir'] = tmp_dir

        # --- Save orion cfg file to tmp ---
        with open(P.join(tmp_dir, '.orion_cfg.yaml'), 'w+') as orion_cfg:
            cfg['orion'].to_yaml(orion_cfg)
            orion_cfg_filepath = orion_cfg.name

        # --- Set Env Variable ---
        set_env_var(cfg)

        # --- Prepare orion command ---
        orion_opt = " "
        exp_opt = " "
        if DEBUG:
            orion_opt += "--debug "
            exp_opt += "--exp-max-trials 1 "
        orion_cmd = (f'orion{orion_opt}hunt -c "{orion_cfg_filepath}" -n "{orion_exp_name}"{exp_opt}'
                     f'python3 run_train.py --config "{cfg_path}"')

        # --- Run orion command ---
        print('>> ', orion_cmd, '\n')
        os.system(orion_cmd)

        # --- Fetch and return run results ---
        tmp_json = P.join(tmp_dir, f'result.json')
        try:
            with open(tmp_json, 'r') as f:
                r = load(f)
            return r
        except OSError as e:
            return {'r_code': -2, 'error': f"{repr(e)} [file={tmp_json}]"}


if __name__ == '__main__':
    main()
