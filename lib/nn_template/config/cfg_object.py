from collections.abc import Mapping
from types import UnionType
from collections.abc import Iterable
import weakref

from .cfg_dict import CfgDict, CfgCollection, CfgList, UNDEFINED, HYPER_PARAMETER
from .hyperparameter_optimization import HyperParameter


class InvalidAttrError(Exception):
    def __init__(self, msg):
        super(InvalidAttrError, self).__init__(msg)


class AttrValueError(Exception):
    def __init__(self, msg, info=''):
        super(AttrValueError, self).__init__(msg + '\n\t' + info)
        self.msg = msg
        self.info = info


class IncompleteObjError(AttrValueError):
    def __init__(self, msg, info=None):
        super(AttrValueError, self).__init__(msg, info)


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
            except (TypeError, AttrValueError):
                raise AttrValueError(f'Incoherent value and type hint for attribute {attr_name}')

            clsdict[attr_name] = attr
            clsdict['__attr__'][attr_name] = attr

        return type.__new__(mcs, name, bases, clsdict)

    def __contains__(cls, item):
        return hasattr(cls, '__attr__') and item in cls.__attr__


def _type2attr(typehint, value=UNDEFINED):
    if isinstance(value, ObjAttr):
        if not issubclass(typehint, ObjCfg):
            raise TypeError
        if value.obj_type is not None and not issubclass(value.obj_type, typehint):
            raise TypeError
        value.obj_type = typehint
        return value
    elif isinstance(value, RefAttr):
        if value.obj_types is not None:
            if all(issubclass(_, typehint) for _ in value.obj_types):
                raise TypeError
        else:
            value.obj_types = typehint.__args__ if isinstance(typehint, UnionType) else typehint
        return value
    elif isinstance(value, CollectionAttr | MultiTypeCollectionAttr | ObjListAttr):
        if issubclass(typehint, CfgCollection):
            raise TypeError
        elif isinstance(typehint, UnionType):
            typehint.__args__
        return value
    elif isinstance(value, OneOfAttr):
        if value.values:
            if all(issubclass(_, typehint) if isinstance(_, type) else isinstance(_, typehint) for _ in value.values):
                raise TypeError
        else:
            value.values = typehint.__args__ if isinstance(typehint, UnionType) else typehint
        return value

    match typehint.__name__:
        case "int": attr = IntAttr
        case "float": attr = FloatAttr
        case "bool": attr = BoolAttr
        case "str": attr = StrAttr
        case "list": attr = lambda default: OneOfAttr(*typehint, default=default)
        case _:
            if isinstance(typehint, ObjCfg):
                attr = lambda default: ObjAttr(obj_type=typehint, default=default)
            elif isinstance(typehint, CfgCollection):
                attr = lambda default: CollectionAttr(obj_types=typehint.obj_types, default=default)
            else:
                attr = Any

    if isinstance(value, CfgAttr):
        if type(value) != attr:
            raise TypeError
        return attr(default=value.default)
    return attr(default=value)


class ObjCfg(CfgDict, metaclass=MetaCfgObj):
    __attr__ = {}

    def __init__(self, data=None, parent=None):
        super(ObjCfg, self).__init__(data=data, parent=parent)
        self._attr_values = {}

    def _init_after_populate(self):
        pass

    def init_after_populate(self):
        # For the ObjAttr not specified in the cfg_dict which should therefore use their default value,
        # create and store the associated CfgObj in _attr_value
        for attr_name, obj_attr in self.attributes(ObjAttr).items():
            if attr_name not in self._attr_values and obj_attr.default is not UNDEFINED:
                self[attr_name] = obj_attr.default

        self._init_after_populate()

        for c in self.attr().values():
            if isinstance(c, ObjCfg):
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
                except AttrValueError as e:
                    raise ParseError(e.msg, mark, e.info) from None
                if isinstance(attr_value, CfgDict):
                    if isinstance(value, CfgDict):
                        value = attr_value
                    else:
                        attr_value.mark = mark
                        attr_value._parent = weakref.ref(self)
                        attr_value._name = key
            self._attr_values[attr_name] = attr_value
        super(ObjCfg, self).__setitem__(key, value)

    def check_integrity(self, recursive=True):
        try:
            attrs = self.attr(default=UNDEFINED)
        except AttrValueError as e:
            from .cfg_parser import ParseError
            raise ParseError(e.msg, self.mark, e.info) from None

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
                if isinstance(attr, ObjCfg):
                    attr.check_integrity(recursive=recursive)
                if isinstance(attr, CfgCollection):
                    for item in attr.values():
                        if isinstance(item, ObjCfg):
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

    def attr(self, default=UNDEFINED):
        return {k: getattr(self, k, default) for k in self.__attr__}

    @classmethod
    def attributes(cls, filter_type=None):
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


class CfgAttr:
    def __init__(self, default=UNDEFINED, nullable=None):
        self.name = ""
        self._checker = None
        if nullable is None:
            nullable = default is None
        self.nullable = nullable
        if default is not None and default is not UNDEFINED:
            default = self.check_value(default)
        self.default = default
        self._parent_cfg_class = None

    def __set_name__(self, owner, name):
        self.name = name
        if name not in owner.__attr__:
            owner.__attr__[name] = self
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
            value = self._checker(value, cfg_dict)
        return self._check_value(value, cfg_dict)

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        return value

    def checker(self, func):
        """
        Check value decorator.
        """
        self._checker = func
        return func

    @property
    def parent_cfg_class(self) -> MetaCfgObj:
        return self._parent_cfg_class

    @property
    def fullname(self):
        return self.parent_cfg_class.__name__ + '.' + self.name


class IntAttr(CfgAttr):
    def __init__(self, default=UNDEFINED, min=None, max=None):
        self.min = min
        self.max = max
        super(IntAttr, self).__init__(default)

    @staticmethod
    def interpret(value) -> int:
        if isinstance(value, str):
            if value.endswith(tuple('TGMk')):
                value = float(value[:-1])*{
                    'T': 1e12, 'G': 1e9, 'M': 1e6, 'k': 1e3
                }[value[-1]]
        return int(value)

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        try:
            value = IntAttr.interpret(value)
        except TypeError:
            raise AttrValueError(f"{value} is not a valid integer for attribute {self.name}")
        if self.min is not None and self.min > value:
            raise AttrValueError(f"Invalid value for attribute {self.name}",
                                 f"Provided value: {value}, exceed the minimum value {self.min}.")
        if self.max is not None and self.max < value:
            raise AttrValueError(f"Invalid value for attribute {self.name}",
                                 f"Provided value: {value}, exceed the maximum value {self.max}.")
        return value


class ShapeAttr(CfgAttr):
    def __init__(self, default=UNDEFINED, dim=None, nullable=None):
        self.dim = dim
        super(ShapeAttr, self).__init__(default, nullable=nullable)

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        try:
            if not isinstance(value, (tuple, list)):
                value = (value,)
                if self.dim is not None:
                    value = value*self.dim
            value = tuple(IntAttr.interpret(_) for _ in value)
        except Exception:
            raise AttrValueError(f"{value} is not a valid integer for attribute {self.name}")

        return value


class FloatAttr(CfgAttr):
    def __init__(self, default=UNDEFINED, min=None, max=None):
        self.min = min
        self.max = max
        super(FloatAttr, self).__init__(default)

    @staticmethod
    def interpret(value) -> float:
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
            raise AttrValueError(f"{value} is not a valid float for attribute {self.name}")

        if self.min is not None and self.min > value:
            raise AttrValueError("Invalid value for attribute {self.name}",
                                 f"Provided value: {value:.4e}, exceed the minimum value {self.min:.4e}.")
        if self.max is not None and self.max < value:
            raise AttrValueError(f"Invalid value for attribute {self.name}",
                                 f"Provided value: {value:.4e}, exceed the maximum value {self.max:.4e}")
        return value


class StrAttr(CfgAttr):
    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        try:
            return str(value)
        except TypeError:
            raise AttrValueError(f"{value} is not a valid string for attribute {self.name}")


class BoolAttr(CfgAttr):
    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        try:
            return bool(value)
        except TypeError:
            from .cfg_parser import format2str
            raise AttrValueError(f"{format2str(value)} is not a valid boolean for attribute {self.name}")


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
            if isinstance(v, CfgAttr):
                try:
                    value = v.check_value(value, cfg_dict=cfg_dict)
                except AttrValueError:
                    pass
                else:
                    break
            elif value == v:
                break
        else:
            raise AttrValueError(f"{value} is invalid for attribute {self.name}",
                                 f"Must be one of {self.values}.")
        return value


class StrMapAttr(CfgAttr):
    def __init__(self, map: Mapping[str, any], default=UNDEFINED):
        self.map = map
        super(StrMapAttr, self).__init__(default)

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        try:
            return self.map[value]
        except KeyError:
            raise AttrValueError(f"{value} is invalid for attribute {self.name} ",
                                 f"Must be one of {list(self.map.keys())}.")


class ObjAttr(CfgAttr):
    def __init__(self, default='__default__', shortcut=None, obj_type=None, nullable=None):
        if obj_type is not None:
            if default == '__default__':
                default = obj_type()
            self.obj_type = obj_type
            if shortcut is not None and (not isinstance(shortcut, str) or shortcut not in obj_type):
                    raise ValueError(f'Invalid shortcut name: {shortcut}. \n'
                                     f'(Valid shortcut are: {list(obj_type.__attr__.keys())}).')
            self.shortcut = shortcut
            super(ObjAttr, self).__init__(default, nullable=nullable)
        else:
            self.shortcut = shortcut
            self.obj_type = None
            super(ObjAttr, self).__init__(nullable=nullable)
            self.default = default

    def __repr__(self):
        return f"Cfg.obj(type={self.obj_type}, shortcut={self.shortcut})"

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        mark = cfg_dict.get_mark(self.name)
        if not isinstance(value, self.obj_type):
            if isinstance(value, Mapping):
                if not isinstance(value, CfgDict):
                    value = CfgDict(value, parent=cfg_dict)
            else:
                if self.shortcut is None:
                    raise AttrValueError(f"{str(value)} is invalid for attribute {self.name}")
                value = CfgDict({self.shortcut: value}, parent=cfg_dict)
        return self.obj_type.from_cfg(value, mark=mark)


class ObjListAttr(CfgAttr):
    def __init__(self, main_key, default=UNDEFINED, obj_types=None):
        self.id_key = main_key
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
            data = value
            value = CfgDict(data={data: data}, parent=cfg_dict)
            value.mark = cfg_dict.child_mark[self.name]
            value.child_mark = {data: cfg_dict.child_mark[self.name]}
        value = CfgList(data=value, obj_types=self.obj_types, id_key=self.id_key, parent=cfg_dict)
        return value


class Any(CfgAttr):
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
            raise AttrValueError(f'Impossible to populate collection {self.fullname}',
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
            raise AttrValueError(f'Impossible to populate collection {self.fullname}',
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
            raise InvalidAttrError(f'Impossible to build the reference attribute {self.fullname}:\n '
                                   f'Unknown path "{self.collection_path}".')
        return collection

    def valid_refs(self, cfg_dict: CfgDict | None = None):
        return set(self.collection(cfg_dict).keys())

    def _check_value(self, name, cfg_dict: CfgDict | None = None):
        if not isinstance(name, str):
            raise AttrValueError(f'Reference key must be string not {type(name)}')

        if cfg_dict is not None:
            referenced_collection = self.collection(cfg_dict)
            if name not in referenced_collection:
                from .cfg_parser import format2str
                raise AttrValueError(f'Unknown reference key "{name}"',
                                     f'Must be one of {format2str(referenced_collection.keys())}.')

        return name

    def __get__(self, instance, owner):
        if instance is None:
            return self

        name = super(RefAttr, self).__get__(instance, owner)
        collection = self.collection(instance)
        if not isinstance(collection, CfgCollection):
            raise InvalidAttrError(f'Impossible to build the reference attribute {self.fullname}\n '
                                   f'Attribute found at path "{self.collection_path}" is not a CfgCollection '
                                   f'but {type(collection).__name__}.')

        if self.obj_types is not None and set(self.obj_types) != set(collection.obj_types):
            raise InvalidAttrError(f'Impossible to build the reference attribute {self.fullname}:\n '
                                   f'The CfgCollection found at {self.collection_path} contains '
                                   f'{collection.obj_types.__name__} instead of {self.obj_types}.')

        try:
            return collection[name]
        except KeyError:
            from .cfg_parser import format2str
            raise AttrValueError(f'Unknown reference key "{name}"',
                                 f'Must be one of {format2str(collection.keys())}.')
