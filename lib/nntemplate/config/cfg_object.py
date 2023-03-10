from typing import Mapping, Iterable, Dict
from copy import copy
from types import UnionType
import weakref

from .cfg_dict import CfgDict, CfgCollection, CfgList, UNDEFINED, UNSPECIFIED, HYPER_PARAMETER
from ..hyperparameters_tuning.generic_optimizer import HyperParameter


class InvalidAttrDeclaration(Exception):
    def __init__(self, msg):
        super(InvalidAttrDeclaration, self).__init__(msg)


class InvalidAttr(Exception):
    def __init__(self, error, info='', mark=None):
        super(InvalidAttr, self).__init__(error + (', ' + str(mark) + '.' if mark is not None else '.')
                                         + ('\n\t' + info.replace('\n', '\n\t') if info else ''))
        self.error = error
        self.info = info
        self.mark = mark


class IncompleteObjError(InvalidAttr):
    def __init__(self, error, info=None, mark=None):
        super(IncompleteObjError, self).__init__(error, info=info, mark=mark)


class MetaCfgObj(type):
    def __new__(mcs, name, bases, clsdict):
        # Create class attribute: __attr__ in CfgObj and inherited classes
        if '__attr__' not in clsdict:
            clsdict['__attr__'] = {}
        # Populate __attr__ with attributes of inherited classes if relevant
        for base in bases:
            try:
                clsdict['__attr__'].update(base.__attr__)
            except AttributeError:
                pass

        # Adds class attribute typed by python annotation as CfgAttr
        for attr_name, attr_type in clsdict.get('__annotations__', {}).items():
            attr_value = clsdict.get(attr_name, UNDEFINED)
            try:
                if attr_value is not UNDEFINED:
                    attr = _type2attr(attr_type, value=attr_value)
                else:
                    attr = _type2attr(attr_type)
            except (TypeError, InvalidAttr):
                raise InvalidAttr(f'Incoherent value and type hint for attribute {attr_name}')

            clsdict[attr_name] = attr
            clsdict['__attr__'][attr_name] = attr

        # Register other attributes:
        for attr_name, attr in clsdict.items():
            if isinstance(attr, CfgAttr) and attr_name not in clsdict['__attr__']:
                clsdict['__attr__'][attr_name] = attr

        return type.__new__(mcs, name, bases, clsdict)

    def __contains__(cls, item):
        return hasattr(cls, '__attr__') and item in cls.__attr__


def _check_type_subclass(typehint, obj_types):
    if isinstance(obj_types, type):
        obj_types = (obj_types,)
    elif isinstance(obj_types, Mapping):
        obj_types = tuple(obj_types.values())

    if isinstance(typehint, UnionType):
        typehints = tuple(typehint.__args__)
        return all(any(issubclass(t, hint) for hint in typehints) for t in obj_types)
    else:
        return all(issubclass(t, typehint) for t in obj_types)


def _type2attr(typehint, value=UNDEFINED):
    if isinstance(value, ObjAttr):
        if not issubclass(typehint, CfgObj):
            raise TypeError
        if value.obj_types is not None:
            if not _check_type_subclass(typehint, value.obj_types):
                raise TypeError
        else:
            if isinstance(typehint, UnionType):
                raise TypeError
            value.obj_types = typehint
        return value
    elif isinstance(value, RefAttr):
        if value.obj_types is not None:
            if _check_type_subclass(typehint, value.obj_types):
                raise TypeError
        else:
            value.obj_types = typehint.__args__ if isinstance(typehint, UnionType) else typehint
        return value
    elif isinstance(value, CollectionAttr | MultiTypeCollectionAttr | ObjListAttr):
        if not issubclass(typehint, CfgCollection):
            raise TypeError
        return value
    elif isinstance(value, OneOfAttr):
        if value.values:
            if all(issubclass(_, typehint) if isinstance(_, type) else isinstance(_, typehint) for _ in value.values):
                raise TypeError
        else:
            value.values = typehint.__args__ if isinstance(typehint, UnionType) else typehint
        return value
    elif value is CfgObj:
        raise InvalidAttrDeclaration("Warning: CfgObj must not be assigned as a class attribute.\n"
                                     'You should probably use the CfgAttr "Cfg.obj()" instead. (I know its confusing...)')

    match typehint.__name__ if isinstance(typehint, type) else typehint:
        case "int": attr = IntAttr
        case "float": attr = FloatAttr
        case "bool": attr = BoolAttr
        case "str": attr = StrAttr
        case "list": attr = lambda default: OneOfAttr(*typehint, default=default)
        case _:
            if isinstance(typehint, CfgObj):
                attr = lambda default: ObjAttr(obj_types=typehint, default=default)
            elif isinstance(typehint, CfgCollection):
                attr = lambda default: CollectionAttr(obj_types=typehint.obj_types, default=default)
            else:
                attr = AnyAttr

    if isinstance(value, CfgAttr):
        if type(value) != attr:
            raise TypeError
        return attr(default=value.default)
    return attr(default=value)


class CfgAttr:
    def __init__(self, default=UNDEFINED, nullable=None):
        self.name = ""
        self._checker = None
        self._post_checker = None
        if nullable is None:
            nullable = default is None
        self.nullable = nullable
        if default is not UNDEFINED:
            try:
                default = self.check_value(default)
            except Exception:
                pass
        self.default = default
        self._parent_cfg_class = None

    def __set_name__(self, owner, name):
        self.name = name
        if not hasattr(owner, name):
            setattr(owner, name, self)
        self._parent_cfg_class = owner

    def __set__(self, instance, value):
        raise AttributeError('Attempt to modify a read-only attribute.')

    def __get__(self, instance, owner):
        if instance is None:
            return self

        try:
            return instance._attr_values[self.name]
        except KeyError as e:
            if self.default is UNDEFINED:
                raise AttributeError(f'Attribute {self.name} not yet specified.') from None
            return self.default

    def check_value(self, value, cfg_dict: CfgDict | None = None):
        if self.nullable and value is None:
            return None

        if isinstance(value, str) and value.startswith("$"):
            return value

        if self._checker is not None:
            try:
                value = self._checker(cfg_dict, value)
            except (TypeError, ValueError):
                raise InvalidAttr(f'Invalid value for attribute {self.fullname}',
                                  f'Provided value was: {repr(value)}')
        value = self._check_value(value, cfg_dict)
        if self._post_checker is not None:
            try:
                value = self._post_checker(cfg_dict, value)
            except (TypeError, ValueError):
                raise InvalidAttr(f'Invalid value for attribute {self.fullname}',
                                     f'Provided value was: {repr(value)}')
        return value

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        return value

    def checker(self, func):
        """
        Check value decorator.
        """
        self._checker = func
        return func

    def post_checker(self, func):
        """
        Check value decorator.
        """
        self._post_checker = func
        return func

    @property
    def parent_cfg_class(self) -> MetaCfgObj | None:
        return getattr(self, '_parent_cfg_class', None)

    @property
    def fullname(self):
        parent = self._parent_cfg_class
        if parent:
            return self.parent_cfg_class.__name__ + '.' + self.name
        else:
            return self.name


class CfgObj(CfgDict, metaclass=MetaCfgObj):
    __attr__ = {}

    def __init__(self, data=None, parent=None):
        super(CfgObj, self).__init__(data=None, parent=parent)
        self._attr_values = {}

        #if data is None:
        #    data = {}
        #data.update({name: attr.default for name, attr in self.attributes().items() if attr.default is not UNDEFINED})
        if data:
            self.update(data)

    def _after_populate(self):
        pass

    def init_after_populate(self):
        # For the ObjAttr not specified in the cfg_dict which should therefore use their default value,
        # create and store the associated CfgObj in _attr_value
        for attr_name, obj_attr in self.attributes(ObjAttr).items():
            if attr_name not in self._attr_values and obj_attr.default is not UNDEFINED:
                self[attr_name] = obj_attr.default

        self._after_populate()

        for c in self.attr().values():
            if getattr(c, 'init_after_populate', None) is not None:
                c.init_after_populate()

    def __setitem__(self, key, value):
        attr_name = key.replace('-', '_')
        if isinstance(attr_name, str) and attr_name in self.__attr__.keys():
            from .cfg_parser import ParseError

            attr = self.__attr__[attr_name]
            try:
                mark = self.get_mark(key)
            except KeyError:
                mark = None

            if isinstance(value, HyperParameter) or (isinstance(value, str) and value.startswith('~')):
                attr_value = HYPER_PARAMETER
            else:
                try:
                    attr_value = attr.check_value(value, cfg_dict=self)
                except InvalidAttr as e:
                    if mark is not None:
                        e.mark = mark
                    raise e
                if isinstance(attr_value, CfgDict):
                    if isinstance(value, CfgDict):
                        value = attr_value
                    else:
                        attr_value.mark = mark
                        attr_value._parent = weakref.ref(self)
                        attr_value._name = key
            self._attr_values[attr_name] = attr_value
        super(CfgObj, self).__setitem__(key, value)

    def check_integrity(self, recursive=True):
        try:
            attrs = self.attr(default=UNDEFINED)
        except InvalidAttr as e:
            from .cfg_parser import ParseError
            raise ParseError(e.error, self.mark, e.info) from None

        missing_keys = {k for k, v in attrs.items() if v is UNDEFINED}
        if missing_keys:
            from .cfg_parser import format2str
            if len(missing_keys) == 1:
                    raise IncompleteObjError(f"Key {format2str(missing_keys)} is missing to {self.name} definition",
                                     f"This attribute is required to parse {self.name} as a {type(self).__name__}.")
            else:
                raise IncompleteObjError(f"Keys {format2str(missing_keys)} are missing to {self.name} definition",
                                     f"Those attribute are required to parse {self.name} as a {type(self).__name__}.")

        if recursive:
            for attr in attrs.values():
                if isinstance(attr, CfgObj):
                    attr.check_integrity(recursive=recursive)
                if isinstance(attr, CfgCollection):
                    for item in attr.values():
                        if isinstance(item, CfgObj):
                            item.check_integrity(recursive=recursive)

    @classmethod
    def from_cfg(cls, cfg_dict: dict[str, any], mark=None, path=None):
        r = cls()
        if mark is not None:
            r.mark = mark
        if path is not None:
            r._name = path.rsplit('.', 1)[-1]
        if isinstance(cfg_dict, CfgDict) and cfg_dict.parent is not None:
            r._parent = weakref.ref(cfg_dict.parent)

        r.update(cfg_dict)
        return r

    def attr(self, default=UNDEFINED) -> Dict[str, any]:
        return {k: getattr(self, k, default) for k in self.__attr__}

    @classmethod
    def attributes(cls, filter_type=None) -> Dict[str, CfgAttr]:
        if filter_type:
            return {name: attr for name, attr in cls.__attr__.items()
                    if isinstance(attr, filter_type)}
        else:
            return cls.__attr__

    def _repr_markdown_(self):
        return "\n".join([f"**{self.name}** _{type(self).__name__}_:\n"]+[f"- {k}: {v}" for k, v in self.attr().items()])


class CfgCollectionType:
    def __init__(self, obj_types, default_key=None):
        self.obj_types = obj_types if isinstance(obj_types, Iterable) else (obj_types,)
        self.default_key = default_key

    def __call__(self, data=None):
        return self.from_cfg(data)

    def from_cfg(self, cfg_dict: CfgDict, mark=None, path=None, parent=None):
        r = CfgCollection.from_dict(data=cfg_dict, obj_types=self.obj_types, default_key=self.default_key,
                                    recursive=True, read_marks=True, parent=parent)
        if mark is not None:
            r.mark = mark
        if path is not None:
            r._name = path.rsplit(',', 1)[-1]
        return r


class IntAttr(CfgAttr):
    def __init__(self, default=UNDEFINED, min=None, max=None, nullable=None):
        self.min = min
        self.max = max
        super(IntAttr, self).__init__(default, nullable=nullable)

    @staticmethod
    def interpret(value, nullable=False) -> int | None:
        if nullable:
            if value is None or value == '':
                return None

        if isinstance(value, str):
            if value.endswith(tuple('TGMk')):
                value = float(value[:-1])*{
                    'T': 1e12, 'G': 1e9, 'M': 1e6, 'k': 1e3
                }[value[-1]]
        return int(value)

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        try:
            value = IntAttr.interpret(value)
        except (TypeError, ValueError):
            raise InvalidAttr(f"{value} is not a valid integer for attribute {self.fullname}")
        if self.min is not None and self.min > value:
            raise InvalidAttr(f"Invalid value for attribute {self.fullname}",
                                 f"Provided value: {value}, exceed the minimum value {self.min}.")
        if self.max is not None and self.max < value:
            raise InvalidAttr(f"Invalid value for attribute {self.fullname}",
                                 f"Provided value: {value}, exceed the maximum value {self.max}.")
        return value


class ShapeAttr(CfgAttr):
    def __init__(self, default=UNDEFINED, dim=None, nullable=None):
        self.dim = dim
        super(ShapeAttr, self).__init__(default, nullable=nullable)

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        try:
            if isinstance(value, str):
                value = value.strip('() \t').split(',')
            if not isinstance(value, (tuple, list)):
                value = (value,)
                if self.dim is not None:
                    value = value*self.dim
            value = tuple(IntAttr.interpret(_) for _ in value)
        except Exception:
            raise InvalidAttr(f"{value} is not a valid integer for attribute {self.fullname}")

        return value


class FloatAttr(CfgAttr):
    def __init__(self, default=UNDEFINED, min=None, max=None, nullable=None):
        self.min = min
        self.max = max
        super(FloatAttr, self).__init__(default, nullable=nullable)

    @staticmethod
    def interpret(value, nullable=False) -> float | None:
        if nullable:
            if value is None or value == '':
                return None

        if isinstance(value, str):
            value = value.strip()
            if value.endswith('%'):
                value = float(value[:-1])/100
            elif value.endswith('‰'):
                value = float(value[:-1])/100
            elif value.endswith(tuple('TGMkmµn')):
                value = float(value[:-1])*{
                    'T': 1e12, 'G': 1e9, 'M': 1e6, 'k': 1e3, 'm': 1e-3, 'µ': 1e-6, 'n': 1e-9
                }[value[-1]]
        return float(value)

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        try:
            value = FloatAttr.interpret(value)
        except TypeError:
            raise InvalidAttr(f"{value} is not a valid float for attribute {self.fullname}")

        if self.min is not None and self.min > value:
            raise InvalidAttr(f"Invalid value for attribute {self.fullname}",
                                 f"Provided value: {value:.4e}, exceed the minimum value {self.min:.4e}.")
        if self.max is not None and self.max < value:
            raise InvalidAttr(f"Invalid value for attribute {self.fullname}",
                                 f"Provided value: {value:.4e}, exceed the maximum value {self.max:.4e}")
        return value


class RangeAttr(CfgAttr):
    def __init__(self, default=UNDEFINED):
        super(RangeAttr, self).__init__(default)

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        invalid_value = InvalidAttr(f"{value} is not a valid range for attribute {self.fullname}")
        try:
            if isinstance(value, str):
                interval = value.split(':')
                if not 0 < len(interval) <= 3:
                    raise invalid_value
                return slice(*[FloatAttr.interpret(_, nullable=True) for _ in interval])
            return slice(FloatAttr.interpret(value, nullable=True))
        except TypeError:
            raise invalid_value


class StrAttr(CfgAttr):
    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        try:
            return str(value)
        except TypeError:
            raise InvalidAttr(f"Invalid string for attribute {self.fullname}",
                                 f"Provided value was: {value}")


class BoolAttr(CfgAttr):

    @staticmethod
    def interpret(value, nullable=False) -> bool | None:
        if nullable:
            if value is None or value == '':
                return None

        if isinstance(value, str):
            return value.strip().lower() == 'true'
        return bool(value)
    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        try:
            return bool(value)
        except TypeError:
            from .cfg_parser import format2str
            raise InvalidAttr(f"{format2str(value)} is not a valid boolean for attribute {self.fullname}")


class OneOfAttr(CfgAttr):
    def __init__(self, *values, default=UNDEFINED):
        self.values = []
        if values:
            for v in values:
                if isinstance(v, type):
                    self.values += [_type2attr(v)]
                else:
                    self.values += [v]
            super(OneOfAttr, self).__init__(default)
        else:
            self.values = None
            self.default = default
            super(OneOfAttr, self).__init__()

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        for v in self.values:
            interpreted_value = copy(value)
            if isinstance(v, CfgAttr):
                try:
                    interpreted_value = v.check_value(interpreted_value, cfg_dict)
                except InvalidAttr:
                    pass
                else:
                    break
            else:
                if type(v) != type(interpreted_value):
                    match v:
                        case int(), float():
                            interpreted_value = FloatAttr.interpret(interpreted_value)
                        case str():
                            interpreted_value = str(interpreted_value)
                        case bool():
                            interpreted_value = BoolAttr.interpret(interpreted_value)
                if interpreted_value == v:
                    break
        else:
            raise InvalidAttr(f"{value} is invalid for attribute {self.fullname}",
                              f"Must be one of {self.values}.")
        return interpreted_value


class StrMapAttr(CfgAttr):
    def __init__(self, map: Mapping[str, any], default=UNDEFINED):
        self.map = map
        super(StrMapAttr, self).__init__(default)

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        try:
            return self.map[value]
        except KeyError:
            raise InvalidAttr(f"{value} is invalid for attribute {self.fullname}",
                                 f"Must be one of {list(self.map.keys())}.")


class StrListAttr(CfgAttr):
    def __init__(self, default=UNDEFINED, nullable=None):
        super().__init__(default=default, nullable=nullable)

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        if isinstance(value, str):
            return [_.strip() for _ in value.split(',') if _.strip()]
        if isinstance(value, Iterable) and all(isinstance(_, str) for _ in value):
            return list(value)
        raise InvalidAttr(f"Invalid strings list for attribute {self.fullname}",
                             f"Provided value was: {value}")


class ListAttr(CfgAttr):
    def __init__(self, obj_type, min_size: int = 1, max_size: int = None, size: int = None,
                 default=UNDEFINED, nullable=None):
        assert isinstance(obj_type, type) or isinstance(obj_type, CfgAttr)
        if size:
            self.min_size = size
            self.max_size = size
        else:
            if min_size < 0 or (max_size is not None and max_size < min_size):
                raise InvalidAttrDeclaration(f'min_size must be positive and max_size must be higher than min_size')
            self.min_size = min_size
            self.max_size = max_size
        self.obj_type = obj_type
        super().__init__(default=default, nullable=nullable)

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        if isinstance(value, str):
            value = value.split(',')
        elif not isinstance(value, Iterable):
            value = (value,)

        if self.min_size and len(value) < self.min_size:
            raise InvalidAttr(f'Invalid size for list {self.fullname}',
                              f'Size must be higher than {self.min_size} but the provided list only have {len(value)} '
                              f'elements.')
        if self.max_size and len(value) > self.max_size:
            raise InvalidAttr(f'Invalid size for list {self.fullname}',
                              f'Size must be lower than {self.max_size} but the provided list have {len(value)} '
                              f'elements.')

        currated_value = []
        for v in value:
            if isinstance(self.obj_type, CfgAttr):
                currated_value += [self.obj_type.check_value(v, cfg_dict)]
            else:
                try:
                    currated_value += [self.obj_type(v)]
                except:
                    raise InvalidAttr(f'Invalid item "{v}" in list for attribute {self.fullname}',
                                      f'Item should be of type {self.obj_type}.') from None
        return currated_value


class ObjAttr(CfgAttr):
    def __init__(self, default=UNDEFINED, shortcut: str = None, obj_types=None, nullable=None):
        """

        :param default:
        :param shortcut: Shortcut allow to create an object of type ``obj_types`` using a single value instead of
        a complete dictionary. In that case the object will be created using the dictionary ``{shortcut: value}``
        If obj_types is a dictionary of types then shortcut is used to customise the type selector
        (instead of the default attribute name "type").
        :param obj_types: A type inheriting ObjCfg (if None the type is inferred from the typehint).
        It may also be a dictionary of types inheriting ObjCfg as values and with names as keys.
        :param nullable:
        """
        self.shortcut = shortcut
        self.obj_types = obj_types
        if obj_types is not None:
            if not isinstance(obj_types, Mapping):
                self._check_shortcut()
            elif self.shortcut is None:
                self.shortcut = 'type'
            # if default is UNSPECIFIED:
            #     default = obj_types()
        super(ObjAttr, self).__init__(default, nullable=nullable)
        # else:
        #     super(ObjAttr, self).__init__(nullable=nullable)
        #     self.default = UNDEFINED if default is UNSPECIFIED else default

    def _check_shortcut(self):
        if self.shortcut is None:
            return True
        obj_types = self.obj_types
        if isinstance(obj_types, type):
            obj_types = (obj_types,)
        elif isinstance(obj_types, Mapping):
            obj_types = tuple(obj_types.values())
        for t in obj_types:
            if self.shortcut not in t:
                raise ValueError(f'Unknown shortcut attribute: {self.shortcut}, in type {t.__name__}.\n')
        return True

    def __repr__(self):
        return f"Cfg.obj(type={self.obj_types}, shortcut={self.shortcut})"

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        mark = cfg_dict.get_mark(self.name)

        # Pre-Format value
        if not isinstance(value, CfgDict):
            if isinstance(value, (Mapping, list)):
                value = CfgDict.from_dict(value, parent=cfg_dict)
            else:
                if self.shortcut is None:
                    raise InvalidAttr(f"{str(value)} is invalid for attribute {self.fullname}")
                value = CfgDict({self.shortcut: value}, parent=cfg_dict)

        obj_type = self.obj_types
        if isinstance(obj_type, Mapping):
            # In the case of multiple types accepted. Infer the correct one from the type selector attribute.
            if not isinstance(value, Mapping):
                raise InvalidAttr(f"{str(value)} is invalid for attribute {self.fullname}")
            if not self.shortcut in value:
                raise InvalidAttr(f"{str(value)} should contain a type selector field named '{self.shortcut}'"
                                     f"to be valid for attribute {self.fullname}")
            selected_type = value[self.shortcut]
            obj_type = self.obj_types.get(selected_type, None)
            if obj_type is None:
                raise InvalidAttr(f"Unkown type '{selected_type}' for attribute {self.fullname}.",
                                     f"Must be one of: {','.join(self.obj_types.keys)}.")

        if not isinstance(value, obj_type):
            return obj_type.from_cfg(value, mark=mark)
        else:
            return value


class ObjListAttr(CfgAttr):
    def __init__(self, main_key, default=UNDEFINED, type_key=None, obj_types=None):
        self.id_key = main_key
        self.type_key = type_key
        if obj_types is not None:
            self._obj_types = obj_types if isinstance(obj_types, Iterable) else (obj_types,)
        else:
            self._obj_types = None
        super(ObjListAttr, self).__init__(default)

    @property
    def obj_types(self):
        return self._obj_types

    def __repr__(self):
        return f"Cfg.obj_list(type={self.obj_types}, main_key={self.id_key})"

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        if isinstance(value, list):
            value = CfgDict.from_list(value, recursive=True)
        elif isinstance(value, str):
            mark = cfg_dict.child_mark.get(self.name, None) if cfg_dict is not None else None
            data = [v.strip() for v in value.split(',') if v.strip()]
            value = CfgDict(data={d: d for d in data}, parent=cfg_dict,
                            mark=mark, child_mark={d: mark for d in data})
        value = CfgList(data=value, obj_types=self.obj_types, shortcut_key=self.id_key, type_key=self.type_key,
                        parent=cfg_dict)
        return value


class AnyAttr(CfgAttr):
    pass


class CollectionAttr(CfgAttr):
    def __init__(self, obj_types, default=UNDEFINED):
        self._obj_types = obj_types if isinstance(obj_types, Iterable) else (obj_types,)
        super(CollectionAttr, self).__init__(default)

    @property
    def obj_types(self):
        return self._obj_types

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        if isinstance(value, list):
            value = CfgDict.from_list(value, recursive=True)
        if not isinstance(value, dict):
            raise InvalidAttr(f'Impossible to populate collection {self.fullname}',
                                 f'The provided value is not a dictionary.')
        value = CfgCollection(data=value, obj_types=self.obj_types)
        return value


class MultiTypeCollectionAttr(CollectionAttr):
    def __init__(self, obj_types: Mapping[str, type], type_key='type', default=UNDEFINED):
        self.type_key = type_key
        super(MultiTypeCollectionAttr, self).__init__(None, default=default)
        self._obj_types = obj_types

    @property
    def obj_types(self):
        return list(self._obj_types.values())

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        if isinstance(value, list):
            value = CfgDict.from_list(value, recursive=True)
        if not isinstance(value, dict):
            raise InvalidAttr(f'Impossible to populate collection {self.fullname}',
                                 f'The provided value is not a dictionary.')
        value = CfgCollection(data=value, obj_types=self._obj_types, type_key=self.type_key)
        return value


class RefAttr(CfgAttr):
    def __init__(self, collection_path, obj_types=None, default=UNDEFINED):
        self.collection_path = collection_path
        self.obj_types = obj_types

        super(RefAttr, self).__init__(default)

    @property
    def obj_types(self):
        return self._obj_types

    @obj_types.setter
    def obj_types(self, obj_types):
        if obj_types is not None:
            self._obj_types = obj_types if isinstance(obj_types, Iterable) else (obj_types,)
        else:
            self._obj_types = None

    def collection(self, cfg_dict: CfgDict) -> CfgDict:
        try:
            if self.collection_path.startswith('.'):     # Relative ref
                collection = cfg_dict[self.collection_path[1:]]
            else:                       # Absolute ref
                collection = cfg_dict.root()[self.collection_path]
        except KeyError:
            raise InvalidAttrDeclaration(f'Impossible to build the reference attribute {self.fullname}:\n '
                                   f'Unknown path "{self.collection_path}".')
        return collection

    def valid_refs(self, cfg_dict: CfgDict | None = None):
        return set(self.collection(cfg_dict).keys())

    def _check_value(self, name, cfg_dict: CfgDict | None = None):
        if not isinstance(name, str):
            raise InvalidAttr(f'Invalid reference for attribute {self.fullname}',
                                 f'Reference key must be string not {type(name)}')

        if cfg_dict is not None:
            referenced_collection = self.collection(cfg_dict)
            if name not in referenced_collection:
                from .cfg_parser import format2str
                raise InvalidAttr(f'Unknown reference key "{name}" for attribute {self.fullname}',
                                     f'Must be one of {format2str(referenced_collection.keys())}.')

        return name

    def __get__(self, instance, owner):
        if instance is None:
            return self

        name = super(RefAttr, self).__get__(instance, owner)
        collection = self.collection(instance)
        if not isinstance(collection, CfgCollection):
            raise InvalidAttrDeclaration(f'Impossible to build the reference attribute {self.fullname}\n '
                                   f'Attribute found at path "{self.collection_path}" is not a CfgCollection '
                                   f'but {type(collection).__name__}.')

        if self.obj_types is not None and set(self.obj_types) != set(collection.obj_types):
            raise InvalidAttrDeclaration(f'Impossible to build the reference attribute {self.fullname}:\n '
                                   f'The CfgCollection found at {self.collection_path} contains '
                                   f'{collection.obj_types.__name__} instead of {self.obj_types}.')

        try:
            return collection[name]
        except KeyError:
            from .cfg_parser import format2str
            raise InvalidAttr(f'Unknown reference key "{name}"',
                                 f'Must be one of {format2str(collection.keys())}.')
