import inspect
import optuna

from .config import Cfg
from .config.cfg_parser import ParseError
from .config.hyperparameter_optimization import HyperParameter, HyperParametersOptimizerEngine, register_hp_optimizer_engine


class OptunaSamplerCfg(Cfg.Obj):
    samplers = {
        'Random': optuna.samplers.RandomSampler,
        'Grid': optuna.samplers.GridSampler,
        'TPE': optuna.samplers.TPESampler,
        'CmaEs': optuna.samplers.CmaEsSampler,
        'NSGAII': optuna.samplers.NSGAIISampler,
        'QMC': optuna.samplers.QMCSampler,
        'MOTPE': optuna.samplers.MOTPESampler,
    }
    type = Cfg.oneOf(*list(samplers.keys()))

    def create(self):
        sampler = OptunaSamplerCfg.samplers[self.type]
        sampler_arg_keys = inspect.signature(sampler).parameters.keys()
        ignored_arg = set(self.keys()).difference(list(sampler_arg_keys) + ['type'])
        if ignored_arg:
            print('WARNING: The following argument(s) for the optuna parser will be ignored:\n\t' + ', '.join(ignored_arg))
        kwargs = {k: v for k, v in self.items() if k in sampler_arg_keys}
        return sampler(**kwargs)


class OptunaPrunerCfg(Cfg.Obj):
    pruners = {
        'Median': optuna.pruners.MedianPruner,
        'Nop': optuna.pruners.NopPruner,
        'Patient': optuna.pruners.PatientPruner,
        'Percentile': optuna.pruners.PercentilePruner,
        'SuccessiveHalving': optuna.pruners.SuccessiveHalvingPruner,
        'Hyperband': optuna.pruners.HyperbandPruner,
        'Threshold': optuna.pruners.ThresholdPruner,
    }
    type = Cfg.oneOf(*list(pruners.keys()))

    def create(self):
        pruner = OptunaPrunerCfg.pruners[self.type]
        pruner_arg_keys = inspect.signature(pruner).parameters.keys()
        ignored_arg = set(self.keys()).difference(list(pruner_arg_keys) + ['type'])
        if ignored_arg:
            print('WARNING: The following argument(s) for the optuna pruner will be ignored:\n\t' + ', '.join(ignored_arg))
        kwargs = {k: v for k, v in self.items() if k in pruner_arg_keys}
        return pruner(**kwargs)


@Cfg.register_obj("optuna")
class OptunaCfg(Cfg.Obj):
    storage = Cfg.str(default=None)
    study_name = Cfg.str(default=None)
    direction = Cfg.oneOf('minimize', 'maximize', default='minimize')
    sampler: OptunaSamplerCfg = Cfg.obj(shortcut='type', default=None, nullable=True)
    pruner: OptunaPrunerCfg = Cfg.obj(shortcut='type', default=None, nullable=True)

    def init_after_populate(self):
        self._engine = OptunaEngine(self.root())
        self._engine.discover_hyperparameters()

    @property
    def engine(self):
        return self._engine

    @property
    def study(self):
        study = getattr(self, '_study', None)
        if study is None:
            study = self.create_study(load_if_exists=True)
            self._study = study
        return study

    @property
    def trial(self):
        return getattr(self, '_trial', None)

    def experiment_study_name(self):
        if self.study_name is not None:
            return self.study_name
        exp_name = self.root().get('experiment.name', None)
        project = self.root().get('experiment.project', None)
        entity = self.root().get('experiment.entity', None)
        study_name = [_ for _ in (entity, project, exp_name) if _]
        if study_name:
            return '/'.join(study_name)
        return None

    def create_study(self, load_if_exists=False):
        kwargs = {}
        if self.sampler:
            kwargs['sampler'] = self.sampler.create()
        if self.pruner:
            kwargs['pruner'] = self.pruner.create()

        storage = self.storage if not self.root().get('hardware.debug', False) else None
        study_name = self.experiment_study_name()

        study = optuna.create_study(study_name=study_name, storage=storage,
                                   direction=self.direction,
                                   load_if_exists=load_if_exists,
                                   **kwargs)
        self._study = study
        return study

    def load_study(self):
        kwargs = {}
        if self.sampler:
            kwargs['sampler'] = self.sampler.create()
        if self.pruner:
            kwargs['pruner'] = self.pruner.create()

        study_name = self.experiment_study_name()
        study = optuna.load_study(study_name=study_name, storage=self.storage, **kwargs)
        self._study = study
        return study

    def get_all_study_summaries(self, include_best_trial=True):
        return optuna.get_all_study_summaries(storage=self.storage, include_best_trial=include_best_trial)

    def copy_study(self, to_storage=None, to_study_name=None):
        if to_storage is None and to_study_name is None:
            raise ValueError('Impossible to copy study, either to_storage or to_study_name must be defined.')
        if to_storage is None:
            to_storage = self.storage
        if to_study_name is None:
            to_study_name = self.study_name

        optuna.copy_study(from_study_name=self.study_name, from_storage=self.storage,
                          to_study_name=to_study_name, to_storage=to_storage)

        kwargs = {}
        if self.sampler:
            kwargs['sampler'] = self.sampler.create()
        if self.pruner:
            kwargs['pruner'] = self.pruner.create()
        return optuna.load_study(storage=to_storage, study_name=to_study_name, **kwargs)

    def ask(self, suggest=True) -> optuna.trial.Trial:
        trial = self.study.ask()
        self.root()['experiment.run_id'] = trial.number
        if suggest:
            self.engine.suggest(trial)
        return trial

    def report(self, optimized_value, step):
        trial = self.trial
        if trial:
            trial.report(optimized_value, step)

    def should_prune(self):
        trial = self.trial
        if trial:
            prune = trial.should_prune()
            if prune:
                self.study.tell(self.trial, state=optuna.trial.TrialState.PRUNED)
            trial = None
            return prune
        return False

    def tell(self, optimized_value):
        trial = self.trial
        if trial:
            self.study.tell(self.trial, optimized_value)
            trial = None

# =====================================================================================================================


@register_hp_optimizer_engine
class OptunaEngine(HyperParametersOptimizerEngine):
    def __init__(self, cfg: Cfg.Dict):
        super(OptunaEngine, self).__init__(cfg)

    def create_hyperparameter(self, name, parent, specification, mark) -> HyperParameter:
        return OptunaHP(name, parent, specification, self, mark)

    def suggest(self, trial: optuna.trial.Trial):
        for hp in self.hyper_parameters.values():
            hp.suggest(trial)
        self.apply_suggestion()

    @classmethod
    def engine_name(cls):
        return "optuna"


class OptunaHP(HyperParameter):
    def __init__(self, name: str, parent: Cfg.Dict, specs: str, engine: OptunaEngine, mark):
        self.name = name
        self._suggested_value = None
        self._type, self.args, self.kwargs = OptunaHP.parse_args(specs.split('.', 1)[-1], mark)
        super(OptunaHP, self).__init__(name, parent, specs, engine, mark)

    @property
    def type(self):
        return {'int': int, 'float': float, 'categorical': None}[self._type]

    @staticmethod
    def parse_args(specs: str, mark=None):
        type, args = specs.split('(', 1)
        args = args.rsplit(')', 1)[0]

        match type:
            case 'int': suggest_f = optuna.trial.Trial.suggest_int
            case 'float': suggest_f = optuna.trial.Trial.suggest_float
            case 'categorical': suggest_f = optuna.trial.Trial.suggest_categorical
            case _:
                from .config.cfg_parser import format2str
                raise ParseError(f"Invalid optuna suggestion method {format2str(type)}\n", mark,
                                 "Valid method are optuna.int, optuna.float or optuna.categorical.")

        def f(*args, **kwargs):
            return args, kwargs
        args = eval(f"f('foo', {args})")
        try:
            args = inspect.signature(suggest_f).bind(None, *args[0], **args[1])
        except (TypeError, AttributeError) as e:
            raise ParseError(f"Invalid argument for ~optuna.{type}", mark, str(e)) from None
        args.apply_defaults()
        return type, args.args[2:], args.kwargs

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

        v = suggest_f(self.name, *self.args, **self.kwargs)
        self.suggested_value = v
        return v


def apply_optuna_suggestion(cfg: Cfg.Dict, trial: optuna.trial.Trial):
    def suggest(key, value):
        if isinstance(value, str) and value.startswith("$optuna."):
            value = OptunaHP(key, value[8:])
            value.suggest(trial)
        elif isinstance(value, OptunaHP):
            value.suggest(trial)
        return value

    cfg.map(suggest, True)


def clear_optuna_suggestion(cfg: Cfg.Dict):
    pass
