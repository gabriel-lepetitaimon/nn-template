from functools import cached_property
import inspect
import optuna
import traceback

import pytorch_lightning as pl

from ..config import Cfg
from ..config.cfg_parser import ParseError
from ..hyperparameters_tuning.generic_optimizer import HyperParameter, HyperParametersOptimizerEngine, \
    register_hp_optimizer_engine
from ..training import TrainingCfg, check_metric_name, MonitoredMetricCfg
from ..hardware import HardwareCfg
from ..callbacks.optuna import PyTorchLightningPruningCallback

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
    monitor: MonitoredMetricCfg = Cfg.obj(None)

    def create(self):
        pruner = OptunaPrunerCfg.pruners[self.type]
        pruner_arg_keys = inspect.signature(pruner).parameters.keys()
        ignored_arg = set(self.keys()).difference(list(pruner_arg_keys) + ['type', 'monitor'])
        if ignored_arg:
            print('WARNING: The following argument(s) for the optuna pruner will be ignored:\n\t' + ', '.join(
                ignored_arg))
        kwargs = {k: v for k, v in self.items() if k in pruner_arg_keys}
        return pruner(**kwargs)

    def _default_monitored_metrics(self) -> MonitoredMetricCfg:
        training_cfg: TrainingCfg = self.root()['training']
        return training_cfg.objective_ckpt_cfg

    @property
    def monitored_metric(self) -> MonitoredMetricCfg:
        return self.monitor if self.monitor is not None else self._default_monitored_metrics()


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
    _optuna_retry = 0

    n_runs = Cfg.int()
    storage: OptunaRDBStorageCfg = Cfg.obj(shortcut='url', default=None)
    study_name = Cfg.str(default=None)
    direction = Cfg.oneOf('minimize', 'maximize', default='minimize')
    sampler: OptunaSamplerCfg = Cfg.obj(shortcut='type', default=None, nullable=True)
    pruner: OptunaPrunerCfg = Cfg.obj(shortcut='type', default=None, nullable=True)
    max_retry = 3

    def init_after_populate(self):
        self._engine = OptunaEngine(self.root())
        self._engine.discover_hyperparameters()

    class TrialContext:
        def __init__(self, optuna_cfg, max_retry):
            self.cfg = optuna_cfg
            self.max_retry = max_retry

        def __enter__(self):
            self.cfg.ask()
            return self.cfg.root()

        def __exit__(self, exc_type, exc_val, exc_tb):
            trial = self.cfg.trial
            if isinstance(trial, optuna.Trial):
                if exc_type is None:
                    self.cfg.study.tell(trial, state=optuna.trial.TrialState.COMPLETE)
                else:
                    self.cfg.study.tell(trial, state=optuna.trial.TrialState.FAIL)
            self.cfg.clear_trial()

            if exc_type is not None:
                if self.cfg.root().get('hardware.debug', False):
                    return

                self.cfg._optuna_retry += 1
                max_retry_reached = self.cfg._optuna_retry >= self.max_retry
                if not max_retry_reached:
                    traceback.print_exc()
                    return "Continue execution despite the exception"
                return


    def trial_ctx(self, max_retry=5):
        return OptunaCfg.TrialContext(self, max_retry=max_retry)

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
        Create or load a study using the information from the configuration
            (namely the proper name, sampler, pruner and storage).
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

    def valid_trials_count(self, only_completed=False):
        trials = self.study.trials
        n_valid_trials = 0
        State = optuna.trial.TrialState
        for trial in trials:
            if (only_completed and trial.state in (State.COMPLETE, State.PRUNED)) \
                    or trial.state != State.FAIL:
                n_valid_trials += 1
        return n_valid_trials

    def hyper_parameter_search_complete(self):
        return self.valid_trials_count() >= self.n_runs

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

    def optimize(self, func, *args, **kwargs):
        if self.hyper_parameter_search_complete():
            return

        hardware: HardwareCfg = self.root()['hardware']
        training: TrainingCfg = self.root()['training']
        debug_run = hardware.debug
        def run_trial(trial):
            self._trial = trial
            self.root()['experiment.trial_id'] = self.valid_trials_count()
            self.engine.suggest(trial)

            try:
                opti_value = func(*args, **kwargs)
            except optuna.TrialPruned as e:
                raise e
            except Exception as e:
                self._optuna_retry += 1
                max_retry_reached = self._optuna_retry >= self.max_retry
                if not max_retry_reached and not debug_run:
                    traceback.print_exc()
                    raise OptunaCfg.IgnoreException() from e
                else:
                    raise e

            if self.hyper_parameter_search_complete() or debug_run:
                self.study.stop()
            if training.objective_ckpt_cfg.mode == 'max':
                opti_value = -opti_value
            return opti_value

        self.study.optimize(run_trial, show_progress_bar=True, catch=OptunaCfg.IgnoreException)
        self.clear_trial()

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

    def pl_callbacks(self) -> list[pl.Callback]:
        """
        Generate a pruning callback
        :return:
        """
        trial = self.trial
        monitor = self.pruner.monitored_metric
        if isinstance(trial, optuna.Trial):
            return [PyTorchLightningPruningCallback(trial, monitor=monitor.metric, mode=monitor.mode)]
        else:
            return []

    def tell(self, optimized_value):
        trial = self.trial
        if isinstance(trial, optuna.Trial):
            self.study.tell(trial, optimized_value)
            self._trial = None

    def clear_trial(self):
        self.engine.clear_suggestion()
        self._trial = None

    def hyper_parameters(self):
        return Cfg.Dict.from_dict({k: hp.suggested_value for k, hp in self.engine.hyper_parameters.items()})

    class IgnoreException(Exception):
        def __init__(self):
            super().__init__("Exception should be ignored.")


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
