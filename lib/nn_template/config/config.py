__all__ = ['default_config', 'parse_config']

import os.path as P
import os
from .cfg_dict import CfgDict
from .cfg_parser import CfgParser, register_obj
from .cfg_object import *


ORION_CFG_PATH = P.join(P.dirname(P.abspath(__file__)), '../../../experimentations/src/config/orion_config.yaml')
GLOBAL_CFG_PATH = P.join(P.dirname(P.abspath(__file__)), '../../../experimentations/src/config/global_config.yaml')
DEFAULT_EXP_CFG_PATH = P.join(P.dirname(P.abspath(__file__)),
                              '../../../experimentations/src/config/default_exp_config.yaml')


def default_config():
    with open(ORION_CFG_PATH, 'r') as f:
        orion_config = CfgDict.from_yaml(f)
    with open(GLOBAL_CFG_PATH, 'r') as f:
        global_config = CfgDict.from_yaml(f)
        global_config['orion'] = orion_config
    with open(DEFAULT_EXP_CFG_PATH, 'r') as f:
        global_config.recursive_update(CfgDict.from_yaml(f))
    return global_config


def parse_config(cfg_file):
    default_cfg = default_config()
    if cfg_file is None:
        return default_cfg
    try:
        with open(cfg_file, 'r') as f:
            exp_config = CfgDict.from_yaml(f)
    except:
        exp_config = CfgDict.from_yaml(cfg_file)

    # --- Preprocess cfg file
    if "sub-experiment" not in exp_config.experiment:
        exp_config.experiment['sub-experiment'] = f'[{exp_config.experiment.tags["exp"]}] {exp_config.experiment.tags.get("sub","")}'
    exp_config = exp_config.filter(lambda k, v: not (isinstance(v, str) and v.startswith('orion~')), recursive=True)

    return default_cfg.recursive_update(exp_config)


def parse_arguments(opt=None, require_config=True):
    import argparse

    # --- PARSE ARGS & ENVIRONNEMENTS VARIABLES ---
    if not opt:
        parser = argparse.ArgumentParser()
        kwargs = {'required': True} if require_config else {'default': os.getenv('TRIAL_CFG', None)}
        parser.add_argument('--config', help='config file with hyper parameters - in yaml format',
                            **kwargs)
        parser.add_argument('--debug', help='Debug trial (not logged into orion)', action='store_true',
                            default=os.getenv('TRIAL_DEBUG', None)=="True")
        parser.add_argument('--gpus', help='list of gpus to use for this trial',
                            default=os.getenv('TRIAL_GPUS', 0))
        parser.add_argument('--tmp-dir', help='Directory where the trial temporary folders will be stored.',
                            default=os.getenv('TRIAL_TMP_DIR', None))
        args = vars(parser.parse_args())
    else:
        args = {'config': opt.get('config'),
                'debug': opt.get('debug', False),
                'gpus': opt.get('gpus', None),
                'tmp-dir': opt.get('tmp-dir', None)}
    args = CfgDict.from_dict(args)

    # --- PARSE CONFIG ---
    cfg = parse_config(args['config'])

    # --- Update trial info ---
    if 'trial' in cfg:
        default_trial = cfg.trial
    else:
        default_trial = {}
    trial = {
        'ID': int(os.getenv('TRIAL_ID', default_trial .get('ID', 0))),
        'version': int(os.getenv('TRIAL_VERSION', default_trial .get('version', 0))),
        'name': os.getenv('TRIAL_NAME', default_trial .get('name', "")),
        'cfg_path': os.getenv('TRIAL_CFG_PATH', default_trial .get('cfg_path', "")),
    }
    cfg['trial'] = CfgDict.from_dict(trial)

    # --- Update scripts arguments ---
    if isinstance(args.gpus, str):
        args['gpus'] = [int(_) for _ in args.gpus.split(',')]

    script_args = cfg['script-arguments']
    for k, v in args.items():
        if v is not None:
            k = {"tmp_dir": "tmp-dir"}.get(k,k)
            script_args[k] = v

    # --- Post-process cfg knowing args ---
    if script_args.debug:
        cfg.training['max-epoch'] = 1
    return cfg


def set_env_var(cfg):
    if 'script-arguments' in cfg:
        script_args = cfg['script-arguments']
        if 'config' in script_args:
            os.environ['TRIAL_CFG'] = script_args['config']
        if 'debug' in script_args:
            os.environ['TRIAL_DEBUG'] = str(script_args['debug'])
        if 'gpus' in script_args:
            gpus = script_args['gpus']
            os.environ['TRIAL_GPUS'] = ','.join(str(gpu) for gpu in gpus) if isinstance(gpus, (tuple, list)) else str(gpus)
        if 'tmp-dir' in script_args:
            os.environ['TRIAL_TMP_DIR'] = script_args['tmp-dir']

    if 'trial' in cfg:
        trial = cfg['trial']
        if 'ID' in trial:
            os.environ['TRIAL_ID'] = str(trial['ID'])
        if 'version' in trial:
            os.environ['TRIAL_VERSION'] = str(trial['version'])
        if 'name' in trial:
            os.environ['TRIAL_NAME'] = trial['name']
        if 'cfg_path' in trial:
            os.environ['TRIAL_CFG_PATH'] = trial['cfg_path']


class Cfg:
    int = IntAttr
    float = FloatAttr
    str = StrAttr
    range = RangeAttr
    oneOf = OneOfAttr
    bool = BoolAttr
    strMap = StrMapAttr
    obj = ObjAttr
    collection = CollectionAttr
    obj_list = ObjListAttr
    multi_type_collection = MultiTypeCollectionAttr
    shape = ShapeAttr
    ref = RefAttr

    Parser = CfgParser
    Obj = ObjCfg
    Collection = CfgCollection
    Dict = CfgDict
    Attr = CfgAttr
    InvalidAttr = AttrValueError

    register_obj = register_obj
