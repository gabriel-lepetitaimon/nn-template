from __future__ import annotations
from .cfg_dict import CfgDict, UNDEFINED
from typing import Dict
import weakref

__engines__ = {}
NOT_REGISTERED = 'Engine not yet registered'


class HyperParametersOptimizerEngine:

    def __init__(self, cfg: CfgDict):
        self.hyper_parameters: Dict[str, HyperParameter] = {}
        self._cfg = weakref.ref(cfg)

    @property
    def cfg(self) -> CfgDict:
        return self._cfg()

    def create_hyperparameter(self, name, parent, specification) -> HyperParameter:
        pass

    def apply_suggestion(self):
        self.cfg.update({k: hp.suggested_value if hp.suggested_value is not UNDEFINED else hp
                         for k, hp in self.hyper_parameters.items()})

    def clear_suggestion(self):
        for hp in self.hyper_parameters.values():
            hp.suggested_value = UNDEFINED
        self.cfg.update(self.hyper_parameters)

    def discover_hyperparameters(self):
        for cursor in self.cfg.walk_cursor():
            if isinstance(cursor.value, str) and cursor.value.startswith('~'+self.engine_name()):
                hp = self.create_hyperparameter(cursor.name, cursor.parent, cursor.value)
                self.hyper_parameters[hp.fullname] = hp
            self.clear_suggestion()

    @classmethod
    def engine_name(cls):
        return "BaseEngine"


class HyperParameter:
    def __init__(self, name, parent: CfgDict, full_specifications: str, engine: HyperParametersOptimizerEngine):
        self.name = name
        self._parent = weakref.ref(parent)
        self._engine = weakref.ref(engine)
        self.full_specifications = full_specifications
        self.suggested_value = UNDEFINED

    @property
    def parent(self):
        return self._parent()

    @property
    def fullname(self):
        return self.parent.fullname + '.' + self.name

    @property
    def engine(self) -> HyperParametersOptimizerEngine:
        return self._engine()


def register_hp_optimizer_engine(engine: type):
    if not issubclass(engine, HyperParametersOptimizerEngine):
        raise RuntimeError(
            f"To be registered, an hyper-parameter optimizer engine must inherit HyperParametersOptimizerEngine.")
    name = engine.engine_name()
    if name in __engines__:
        raise RuntimeError(
            f'An Hyper-Parameters Optimizer Engine is already registered under the name "{name}".')
    __engines__[name] = engine
    return engine

