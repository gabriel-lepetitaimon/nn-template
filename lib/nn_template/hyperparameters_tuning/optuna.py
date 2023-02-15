import inspect
import optuna
import pytorch_lightning
from functools import cached_property

from nn_template.config import Cfg
from nn_template.config.cfg_parser import ParseError
from nn_template.hyperparameters_tuning.generic_optimizer import HyperParameter, HyperParametersOptimizerEngine, \
    register_hp_optimizer_engine
from nn_template.training import TrainingCfg, check_metric_name

from typing import List


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
            print('WARNING: The following argument(s) for the optuna parser will be ignored:\n\t' + ', '.join(
                ignored_arg))
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
    monitor = Cfg.str(None)

    @monitor.post_checker
    def monitor_check(self, value: str|None):
        if value is None:
            value = self._default_monitored_metrics()
        check_metric_name(self, value, 'monitor')
        if not value.startswith(('train', 'val')):
            raise Cfg.InvalidAttr(f'Invalid pruner monitored metric: "{value}"',
                                  'Pruner can only monitor metrics computed on the train or the validation dataset.')

    def create(self):
        pruner = OptunaPrunerCfg.pruners[self.type]
        pruner_arg_keys = inspect.signature(pruner).parameters.keys()
        ignored_arg = set(self.keys()).difference(list(pruner_arg_keys) + ['type', 'monitor'])
        if ignored_arg:
            print('WARNING: The following argument(s) for the optuna pruner will be ignored:\n\t' + ', '.join(
                ignored_arg))
        kwargs = {k: v for k, v in self.items() if k in pruner_arg_keys}
        return pruner(**kwargs)

    def _default_monitored_metrics(self):
        training_cfg: TrainingCfg = self.root()['training']
        return training_cfg.objective


class OptunaRDBStorageCfg(Cfg.Obj):
    url = Cfg.str(nullable=True)
    engine_kwargs = Cfg.Dict(None)
    heartbeat_interval = Cfg.int(60, min=0)
    grace_period = Cfg.int(0, min=0)

    def create(self) -> optuna.storages.RDBStorage | None:
        if self.url is None:
            return None
        kwargs = {}
        if self.heartbeat_interval:
            kwargs['heartbeat_interval'] = self.heartbeat_interval
            if self.grace_period:
                kwargs['grace_period'] = self.grace_period
        return optuna.storages.RDBStorage(url=self.url, engine_kwargs=self.engine_kwargs, **kwargs)


@Cfg.register_obj("optuna")
class OptunaCfg(Cfg.Obj):
    storage: OptunaRDBStorageCfg = Cfg.obj(shortcut='url', default=None)
    study_name = Cfg.str(default=None)
    direction = Cfg.oneOf('minimize', 'maximize', default='minimize')
    sampler: OptunaSamplerCfg = Cfg.obj(shortcut='type', default=None, nullable=True)
    pruner: OptunaPrunerCfg = Cfg.obj(shortcut='type', default=None, nullable=True)

    def init_after_populate(self):
        self._engine = OptunaEngine(self.root())
        self._engine.discover_hyperparameters()

    def __enter__(self):
        self.ask()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        trial = self.trial
        if isinstance(trial, optuna.Trial):
            if exc_type is None:
                self.study.tell(trial, state=optuna.trial.TrialState.COMPLETE)
            else:
                self.study.tell(trial, state=optuna.trial.TrialState.FAIL)

    @property
    def engine(self):
        return self._engine

    @property
    def study(self) -> optuna.Study:
        """
        The current optuna study. The study is loaded or created using the configuration information on the first time
        this method is called.
        """
        study = getattr(self, '_study', None)
        if study is None:
            study = self.create_study(load_if_exists=True)
            self._study = study
        return study

    @cached_property
    def optuna_storage(self):
        storage = self.storage if not self.root()['hardware'].debug else None
        return storage.create() if storage is not None else None

    @property
    def trial(self) -> optuna.trial.BaseTrial | None:
        """
        The current optuna trial. Maybe None if  neither ask() nor load_trial() was called.
        """
        return getattr(self, '_trial', None)

    def experiment_study_name(self):
        from ..experiment import ExperimentCfg
        experiment: ExperimentCfg = self.root()['experiment']
        study_name = self.study_name
        if study_name is None:
            study_name = '/'.join([_ for _ in
                                   (experiment.project, experiment.name) if _])
        return study_name + '--' + experiment.experiment_hash

    def create_study(self, load_if_exists=False) -> optuna.Study:
        """
        Create or load a study using the information from the configuration (namely the proper name, sampler, pruner and storage).
        The created or loaded study is stored in ``self.study``.
        :param load_if_exists: Flag to control the behavior to handle a conflict of study names.
            In the case where a study named ``study_name`` already exists in the ``storage``,
            a :class:`~optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is
            set to :obj:`False`.
            Otherwise, the creation of the study is skipped, and the existing one is returned.
        :return: The created or loaded study.
        """
        kwargs = {}
        if self.sampler:
            kwargs['sampler'] = self.sampler.create()
        if self.pruner:
            kwargs['pruner'] = self.pruner.create()

        study = optuna.create_study(study_name=self.experiment_study_name(), storage=self.optuna_storage,
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
        study = optuna.load_study(study_name=study_name, storage=self.optuna_storage, **kwargs)
        self._study = study
        return study

    def get_all_study_summaries(self, include_best_trial=True) -> List[optuna.study.StudySummary]:
        return optuna.get_all_study_summaries(storage=self.optuna_storage, include_best_trial=include_best_trial)

    def valid_trials_count(self):
        trials = self.study.trials
        n_valid_trials = 0
        for trial in trials:
            if trial.state != optuna.trial.TrialState.FAIL:
                n_valid_trials += 1
        return n_valid_trials

    def copy_study(self, to_storage=None, to_study_name=None):
        if to_storage is None and to_study_name is None:
            raise ValueError('Impossible to copy study, either to_storage or to_study_name must be defined.')
        if to_storage is None:
            to_storage = self.optuna_storage
        if to_study_name is None:
            to_study_name = self.study_name

        optuna.copy_study(from_study_name=self.study_name, from_storage=self.optuna_storage,
                          to_study_name=to_study_name, to_storage=to_storage)

        kwargs = {}
        if self.sampler:
            kwargs['sampler'] = self.sampler.create()
        if self.pruner:
            kwargs['pruner'] = self.pruner.create()
        return optuna.load_study(storage=to_storage, study_name=to_study_name, **kwargs)

    def ask(self, suggest=True) -> optuna.trial.Trial:
        """
        Create a new trial optuna.
        :param suggest: If :obj:`True` generate values for all the hyperparameters and apply them to the CfgDict.
        :return:
        """
        trial = self.trial
        if isinstance(trial, optuna.Trial):
            self.study.tell(trial, state=optuna.trial.TrialState.FAIL)

        trial = self.study.ask()
        self._trial = trial
        self.root()['experiment.run_id'] = trial.number
        if suggest:
            self.engine.suggest(trial)
        return trial

    def _load_trial(self, trial: optuna.trial.FrozenTrial, merge=False):
        previous_trial = self.trial
        if isinstance(previous_trial, optuna.Trial):
            self.study.tell(previous_trial, state=optuna.trial.TrialState.FAIL)

        self._trial = trial
        self.engine.force_hyperparameters_value(self.trial.params, merge=merge)

    def load_trial(self, id=None, merge=False) -> optuna.trial.BaseTrial:
        if self.trial is not None and self.trial.number == id:
            return self.trial

        trial = self.study.trials[id]
        self._load_trial(trial, merge=merge)
        return trial

    def load_best_trial(self, merge=False) -> optuna.trial.FrozenTrial:
        trial = self.study.best_trial
        if trial is not None:
            self._load_trial(trial, merge=merge)
        return trial

    def report(self, optimized_value, step):
        """
        Report an ``optimized_value`` to the current trial for a given ``step``.
        """
        trial = self.trial
        if isinstance(trial, optuna.Trial):
            trial.report(optimized_value, step)

    def should_prune(self):
        """
        Check if the current trial should be pruned or not according to the study pruner.
        If it's the case, the trial **is** pruned and ``self.trial`` is set to :obj:`None`.
        :return: :obj:`True` if the trial has been pruned, :obj:`False` otherwise.
        """
        trial = self.trial
        if isinstance(trial, optuna.Trial):
            prune = trial.should_prune()
            if prune:
                self.study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                self._trial = None
            return prune
        return False

    def pytorch_lightnings_callbacks(self) -> list[optuna.integration.PyTorchLightningPruningCallback]:
        """
        Generate a pruning callback
        :param monitor:
        :return:
        """
        trial = self.trial
        monitor = self.pruner.monitor
        if monitor is None:
            monitor = self.root()['training'].objective
        if isinstance(trial, optuna.Trial):
            return [optuna.integration.pytorch_lightning.PyTorchLightningPruningCallback(trial, monitor)]
        else:
            return []

    def tell(self, optimized_value):
        trial = self.trial
        if isinstance(trial, optuna.Trial):
            self.study.tell(trial, optimized_value)
            self._trial = None


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
            case 'int':
                suggest_f = optuna.trial.Trial.suggest_int
            case 'float':
                suggest_f = optuna.trial.Trial.suggest_float
            case 'categorical':
                suggest_f = optuna.trial.Trial.suggest_categorical
            case _:
                from ..config.cfg_parser import format2str
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
