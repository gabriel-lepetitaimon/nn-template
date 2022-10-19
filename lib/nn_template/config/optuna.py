import optuna
import inspect
from .config import Cfg


class OptunaHP:
    def __init__(self, name:str, desc: str):
        self.name = name
        self._type, self.args = OptunaHP.parse_args(desc)

    @property
    def type(self):
        return {'int': int, 'float': float, 'categorical': None}[self._type]

    @staticmethod
    def parse_args(name: str, desc: str):
        type, args = desc.split('(', 1)
        args = args.rsplit(')', 1)[0]

        match type:
            case 'int': suggest_f = optuna.trial.Trial.suggest_int
            case 'float': suggest_f = optuna.trial.Trial.suggest_float
            case 'categorical': suggest_f = optuna.trial.Trial.suggest_categorical
            case _: raise ValueError(f"Invalid type for Optuna Hyper Parameter suggestion.\n"
                                     f"Valid types are int, float or categorical, provided type is '{type}'.")

        def f(*args, **kwargs):
            return args, kwargs
        args = eval(f"f('test', {args})")
        args = inspect.signature(suggest_f).bind(*args[0])
        args.apply_defaults()
        return type, args

    def suggest(self, trial: optuna.trial.Trial):
        match self._type:
            case 'int':
                suggest_f = trial.suggest_int
            case 'float':
                suggest_f = trial.suggest_float
            case 'categorical':
                suggest_f = trial.suggest_categorical
            case _:
                return
        return suggest_f(self.name, *self.args.args, **self.args.kwargs)


def suggest_optuna(cfg: Cfg.Dict, trial: optuna.trial.Trial):
    def parse_optuna(key, value):
        if isinstance(value, str) and value.startswith("$optuna."):
            return OptunaHP(key, value[8:]).suggest(trial)
        return value

    cfg.map(parse_optuna, True)
