import weakref
from collections.abc import Mapping
from types import UnionType

from .cfg_dict import CfgDict, CfgCollection


UNDEFINED = '__undefined__'


class InvalidCfgAttr(Exception):
    def __init__(self, msg):
        super(InvalidCfgAttr, self).__init__(msg)


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
            except TypeError:
                raise InvalidCfgAttr(f'Incoherent value and type hint for attribute {attr_name}.')

            attr.name = attr_name
            clsdict['__attr__'][attr_name] = attr

        return type.__new__(mcs, name, bases, clsdict)

    def __contains__(cls, item):
        return hasattr(cls, '__attr__') and item in cls.__attr__


def _type2attr(typehint, value=UNDEFINED):
    if issubclass(typehint, CfgObj):
        if isinstance(value, ObjAttr):
            return ObjAttr(type=typehint, shortcut=value.shortcut, default=value.default)
        else:
            return ObjAttr(type=typehint, default=value)
    elif isinstance(typehint, UnionType):
        if isinstance(value, OneOfAttr):
            if value.values:
                raise TypeError
            return OneOfAttr(typehint.__args__, default=value.default)
        elif isinstance(value, ObjAttr):
            raise TypeError
        else:
            return OneOfAttr(typehint.__args__, default=value)

    match typehint.__name__:
        case "int": typehint = IntAttr
        case "float": typehint = FloatAttr
        case "bool": typehint = BoolAttr
        case "str": typehint = StrAttr
        case _: typehint = Any

    if isinstance(value, CfgAttr):
        if type(value) != typehint:
            raise TypeError
        return typehint(default=value.default)
    return typehint(default=value)


class CfgObj(CfgDict, metaclass=MetaCfgObj):
    __attr__ = {}

    def __init__(self, data=None, parent=None):
        super(CfgObj, self).__init__(data=data, parent=parent)
        self._attr_values = {}

    def __setitem__(self, key, value):
        attr_name = key.replace('-', '_')
        if isinstance(attr_name, str) and attr_name in self.__attr__.keys():
            from .cfg_parser import ParseError

            attr = self.__attr__[attr_name]
            try:
                mark = self.get_mark(key)
            except KeyError:
                mark = None

            try:
                attr_value = attr.check_value(value)
            except (InvalidCfgAttr, ParseError) as e:
                raise ParseError(str(e), mark) from None
            if isinstance(attr_value, CfgDict):
                if isinstance(value, CfgDict):
                    value = attr_value
                else:
                    attr_value.mark = mark
                    attr_value._parent = weakref.ref(self)
                    attr_value._name = key
            self._attr_values[attr_name] = attr_value
        super(CfgObj, self).__setitem__(key, value)

    def check_integrity(self):
        missing_keys = {k for k, v in self.attr(default=UNDEFINED).items() if v is UNDEFINED}
        if missing_keys:
            raise InvalidCfgAttr(f"Missing not-optional attributes {tuple(missing_keys)} "
                                 f"to define {type(self).__name__}")

    @classmethod
    def from_cfg(cls, cfg_dict: dict[str, any], mark=None):
        r = cls()
        r.update(cfg_dict)
        if mark is not None:
            r.mark = mark
        try:
            r.check_integrity()
        except InvalidCfgAttr as e:
            from .cfg_parser import ParseError
            raise ParseError(str(e), mark) from None
        return r

    def attr(self, default=UNDEFINED):
        return {k: getattr(self, k, default) for k in self.__attr__}

    def _repr_markdown_(self):
        return "\n".join([f"**{self.name}** _{type(self).__name__}_:\n"]+[f"- {k}: {v}" for k, v in self.attr().items()])


class CfgCollectionType:
    def __init__(self, obj_type, default_key=None):
        self.obj_type = obj_type
        self.default_key = default_key

    def __call__(self, data=None):
        return self.from_cfg(data)

    def from_cfg(self, cfg_dict: CfgDict, mark=None):
        r = CfgCollection.from_dict(data=cfg_dict, obj_type=self.obj_type, default_key=self.default_key,
                                    recursive=True, read_marks=True)
        if mark is not None:
            r.mark = mark
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
        self._cfg_obj = None

    def __set_name__(self, owner, name):
        self.name = name
        if name not in owner.__attr__:
            owner.__attr__[name] = self
        self._cfg_obj = weakref.ref(owner)

    def __set__(self, instance, value):
        raise AttributeError('Attempt to modify a read-only attribute.')

    def __get__(self, instance, owner):
        try:
            return instance._attr_values[self.name]
        except KeyError as e:
            if self.default is UNDEFINED:
                raise AttributeError(f'Attribute {self.name} not yet specified.') from None
            return self.default

    def check_value(self, value):
        if self.nullable and value is None:
            return None
        if self._checker is not None:
            value = self._checker(value)
        return self._check_value(value)

    def _check_value(self, value):
        return value

    def checker(self, func):
        """
        Check value decorator.
        """
        self._checker = func
        return func

    @property
    def cfg_obj(self) -> CfgObj:
        return self._cfg_obj()

    def fullname(self):
        return '.'.join(self.cfg_obj.path() + (self.cfg_obj.name, self.name))


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

    def _check_value(self, value):
        try:
            value = IntAttr.interpret(value)
        except TypeError:
            raise InvalidCfgAttr(f"{value} is not a valid integer for attribute {self.name}")
        if self.min is not None and self.min > value:
            raise InvalidCfgAttr(f"Provided value: {value}, exceed the minimum value {self.min} "
                                 f"for attribute {self.name}")
        if self.max is not None and self.max < value:
            raise InvalidCfgAttr(f"Provided value: {value}, exceed the maximum value {self.max} "
                                 f"for attribute {self.name}")
        return value


class ShapeAttr(CfgAttr):
    def __init__(self, default=UNDEFINED, dim=None):
        self.dim = dim
        super(ShapeAttr, self).__init__(default)

    def _check_value(self, value):
        try:
            if not isinstance(value, (tuple, list)):
                value = (value,)
                if self.dim is not None:
                    value = value*self.dim
            value = tuple(IntAttr.interpret(_) for _ in value)
        except Exception:
            raise InvalidCfgAttr(f"{value} is not a valid integer for attribute {self.name}")

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
            elif value.endswith('???'):
                value = float(value[:-1])/100
            elif value.endswith(tuple('TGMkm??n')):
                value = float(value[:-1])*{
                    'T': 1e12, 'G': 1e9, 'M': 1e6, 'k': 1e3, 'm': 1e-3, '??': 1e-6, 'n': 1e-9
                }[value[-1]]
        return float(value)

    def _check_value(self, value):
        try:
            value = FloatAttr.interpret(value)
        except TypeError:
            raise InvalidCfgAttr(f"{value} is not a valid float for attribute {self.name}")

        if self.min is not None and self.min > value:
            raise InvalidCfgAttr(f"Provided value: {value:.4e}, exceed the minimum value {self.min} "
                                 f"for attribute {self.name}")
        if self.max is not None and self.max < value:
            raise InvalidCfgAttr(f"Provided value: {value:.4e}, exceed the maximum value {self.max} "
                                 f"for attribute {self.name}")
        return value


class StrAttr(CfgAttr):
    def _check_value(self, value):
        try:
            return str(value)
        except TypeError:
            raise InvalidCfgAttr(f"{value} is not a valid string for attribute {self.name}")


class BoolAttr(CfgAttr):
    def _check_value(self, value):
        try:
            return bool(value)
        except TypeError:
            raise InvalidCfgAttr(f"{value} is not a valid boolean for attribute {self.name}")


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

    def _check_value(self, value):
        for v in self.values:
            if isinstance(v, CfgAttr):
                try:
                    value = v.check_value(value)
                except InvalidCfgAttr:
                    pass
                else:
                    break
            elif value == v:
                break
        else:
            raise InvalidCfgAttr(f"{value} is invalid for attribute {self.name} (should be one of {self.values})")
        return value


class StrMapAttr(CfgAttr):
    def __init__(self, map: Mapping[str, any], default=UNDEFINED):
        self.map = map
        super(StrMapAttr, self).__init__(default)

    def _check_value(self, value):
        try:
            return self.map[value]
        except KeyError:
            raise InvalidCfgAttr(f"{value} is invalid for attribute {self.name} "
                                 f"(should be one of {list(self.map.keys())})")


class ObjAttr(CfgAttr):
    def __init__(self, type=None, shortcut=None, default='__default__'):
        if type is not None:
            if default == '__default__':
                default = type()
            self.type = type
            if shortcut is not None and (not isinstance(shortcut, str) or shortcut not in type):
                    raise ValueError(f'Invalid shortcut name: {shortcut}. \n'
                                     f'(Valid shortcut are: {list(type.__attr__.keys())}).')
            self.shortcut = shortcut
            super(ObjAttr, self).__init__(default)
        else:
            self.shortcut = shortcut
            self.type = None
            super(ObjAttr, self).__init__()
            self.default = default

    def __repr__(self):
        return f"Cfg.obj(type={self.type}, shortcut={self.shortcut})"

    def _check_value(self, value):
        if isinstance(value, Mapping):
            value = self.type.from_cfg(value)
        elif not isinstance(value, self.type):
            if self.shortcut is None:
                raise InvalidCfgAttr(f"{value} is invalid for attribute {self.name}.")
            obj = self.type()
            obj[self.shortcut] = value
            value = obj
        return value


class Any(CfgAttr):
    pass


class CollectionAttr(CfgAttr):
    def __init__(self, obj_type, default_key=None, default=UNDEFINED):
        self.obj_type = obj_type
        self.default_key = default_key
        super(CollectionAttr, self).__init__(default)

    def _check_value(self, value):
        if isinstance(value, list):
            value = CfgDict.from_list(value, recursive=True)
        if isinstance(value, Mapping):
            single_value = False
            if self.default_key:
                try:
                    value = self.obj_type.from_cfg(value)
                except (TypeError, InvalidCfgAttr):
                    pass
                else:
                    value = {self.default_key: value}
                    single_value = True
            if not single_value:
                value = {k: self.obj_type.from_cfg(v) for k, v in value.items()}
            value = CfgCollection(data=value, obj_type=self.obj_type, default_key=self.default_key)
        return value

    def _to_obj_type(self, value):
        if issubclass(self.obj_type, CfgObj) or isinstance(self.obj_type, CfgCollectionType):
            return self.obj_type.from_cfg(value)
        else:
            return self.obj_type(value)


class MultiTypeCollectionAttr(CollectionAttr):
    def __init__(self, obj_types: Mapping[str, type], type_key='type', default_key=None, default=UNDEFINED):
        self.type_key = type_key
        self.obj_types = obj_types
        super(MultiTypeCollectionAttr, self).__init__(None, default_key=default_key, default=default)

    def _to_obj_type(self, value):
        if not isinstance(value, dict) or self.type_key not in value:
            raise InvalidCfgAttr(f"Invalid value for {self.name}, "
                                 f"should be a dictionary containing the field {self.type_key}")
        obj_type = self.obj_types.get(value[self.type_key], None)
        if obj_type is None:
            raise InvalidCfgAttr(f"Invalid value for {self.type_key}, "
                                 f"should one of {set(self.obj_type.keys())}")

        self.obj_type = obj_type
        return super(MultiTypeCollectionAttr, self)._to_obj_type(value)


class CollectionRefAttr(CfgAttr):
    def __init__(self, collection_path, default=UNDEFINED):
        self.collection_path = collection_path
        super(CollectionRefAttr, self).__init__(default)

    @property
    def collection(self) -> CfgCollection:
        try:
            if self.collection_path.startswith('.'):     # Relative ref
                collection = self.cfg_obj[self.collection_path[1:]]
            else:                       # Absolute ref
                collection = self.cfg_obj.root()[self.collection_path]
        except KeyError:
            raise InvalidCfgAttr(f'Impossible to build the reference attribute {self.fullname()}:\n '
                                 f'Unknown path "{self.collection_path}".')
        if not isinstance(collection, CfgCollection):
            raise InvalidCfgAttr(f'Impossible to build the reference attribute {self.fullname()}:\n '
                                 f'Attribute found at path "{self.collection_path}" is not a CfgCollection '
                                 f'but {type(collection).__name__}.')
        return collection

    def valid_refs(self):
        return set(self.collection.keys())

    def _check_value(self, value):
        if not isinstance(value, str):
            raise InvalidCfgAttr(f'Reference key must be string not {type(value)}.')
        valid_refs = self.valid_refs()
        if value not in valid_refs:
            raise InvalidCfgAttr(f'Unknown reference key "{value}". \nMust be one of {valid_refs}.')
        return self.collection[value]
