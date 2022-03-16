from collections.abc import Mapping
from types import Union

from .cfg_dict import CfgDict


class InvalidCfgAttr(Exception):
    def __init__(self, msg):
        super(InvalidCfgAttr, self).__init__(msg)


class MetaCfgObj:
    def __new__(cls, *args, **kwargs):
        super(MetaCfgObj, cls).__new__(*args, **kwargs)
        if not hasattr(cls, '__attr__'):
            cls.__attr__ = {}

        for attr_name, attr_type in cls.__annotations__.items():
            attr_value = getattr(cls, attr_name, '__undefined__')
            default = '__undefined__' if isinstance(attr_value, CfgAttr) is None else attr_value
            attr_type = _type2attr(attr_type, default=default)

            if attr_value == '__undefined__':
                attr = attr_type
            elif not isinstance(attr_value, type(attr_type)):
                raise InvalidCfgAttr('Incoherent attribute and type hint.')
            elif isinstance(attr_value, Obj):
                if attr_value.type is not None:
                    if not issubclass(attr_type.type, attr_value.type):
                        raise InvalidCfgAttr('Incoherent attribute and type hint.')
                    attr = attr_value
                else:
                    attr = Obj(type=attr_type.type, shortcut=attr_value.shortcut, default=attr_value.default)
                    attr.name = attr_name
            elif isinstance(attr_value, OneOf):
                if attr_value.values is not None:
                    raise InvalidCfgAttr('Incoherent attribute and type hint.')
                else:
                    attr = OneOf(*attr_type.values, default=attr_value.default)
                    attr.name = attr_name
            else:
                attr = attr_value
            cls.__attr__[attr_name] = attr

    def __contains__(cls, item):
        return item in cls.__attr__


class CfgObj(CfgDict, metaclass=MetaCfgObj):
    __attr__ = {}

    def __init__(self):
        super(CfgObj, self).__init__()
        self._attr_values = {}

    def __setitem__(self, key, value):
        attr_name = key.replace('-', '_')
        if isinstance(attr_name, str) and attr_name in self.__attr__.keys():
            attr = self.__attr__[attr_name]
            try:
                attr_value = attr.check_value(value)
            except InvalidCfgAttr as e:
                from .cfg_parser import ParseError
                try:
                    mark = self.get_mark(key)
                except KeyError:
                    mark = None
                raise ParseError(str(e), mark) from None
            if isinstance(attr_value, CfgDict) and isinstance(value, CfgDict):
                value = attr_value
            self._attr_values[attr_name] = attr_value
        super(CfgObj, self).__setitem__(key, value)

    @classmethod
    def from_cfg(cls, cfg_dict: CfgDict):
        r = cls()
        r.update(cfg_dict)
        return r


class CfgAttr:
    def __init__(self, default='__undefined__'):
        self.name = ""
        if default is not None and default != '__undefined__':
            default = self.check_value(default)
        self.default = default

    def __set_name__(self, owner, name):
        self.name = name
        self._checker = None
        owner.__attr__[name] = self

    def __set__(self, instance, value):
        raise AttributeError('Attempt to modify a read-only attribute.')

    def __get__(self, instance, owner):
        try:
            return instance._attr_values[self.name]
        except KeyError as e:
            if self.default == '__undefined__':
                raise AttributeError(f'Attribute {self.name} not yet specified.') from None

    def check_value(self, value):
        return value if self._checker is None else self.checker(value)

    def checker(self, func):
        """
        Check value decorator.
        """
        self._checker = func
        return func


class Int(CfgAttr):
    def __init__(self, default='__undefined__', min=None, max=None):
        super(Int, self).__init__(default)
        self.min = min
        self.max = max

    @staticmethod
    def interpret(value) -> int:
        if isinstance(value, str):
            if value.endswith(tuple('TGMk')):
                value = float(value[:-1])*{
                    'T': 1e12, 'G': 1e9, 'M': 1e6, 'k': 1e3
                }[value[-1]]
        return int(value)

    def check_value(self, value):
        value = super(Int, self).check_value(value)
        try:
            value = Int.interpret(value)
        except TypeError:
            raise InvalidCfgAttr(f"{value} is not a valid integer for attribute {self.name}")
        if self.min is not None and self.min > value:
            raise InvalidCfgAttr(f"Provided value: {value}, exceed the minimum value {self.min} "
                                 f"for attribute {self.name}")
        if self.max is not None and self.max < value:
            raise InvalidCfgAttr(f"Provided value: {value}, exceed the maximum value {self.max} "
                                 f"for attribute {self.name}")
        return value


class Shape(CfgAttr):
    def __init__(self, default='__undefined__', dim=None):
        super(Shape, self).__init__(default)
        self.dim = dim

    def check_value(self, value):
        value = super(Shape, self).check_value(value)
        try:
            if not isinstance(value, (tuple, list)):
                value = (value,)
                if self.dim is not None:
                    value = value*self.dim
            value = tuple(Int.interpret(_) for _ in value)
        except Exception:
            raise InvalidCfgAttr(f"{value} is not a valid integer for attribute {self.name}")

        return value


class Float(CfgAttr):
    def __init__(self, default='__undefined__', min=None, max=None):
        super(Float, self).__init__(default)
        self.min = min
        self.max = max

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

    def check_value(self, value):
        value = super(Float, self).check_value(value)
        try:
            value = Float.interpret(value)
        except TypeError:
            raise InvalidCfgAttr(f"{value} is not a valid float for attribute {self.name}")

        if self.min is not None and self.min > value:
            raise InvalidCfgAttr(f"Provided value: {value:.4e}, exceed the minimum value {self.min} "
                                 f"for attribute {self.name}")
        if self.max is not None and self.max < value:
            raise InvalidCfgAttr(f"Provided value: {value:.4e}, exceed the maximum value {self.max} "
                                 f"for attribute {self.name}")
        return value

class Str(CfgAttr):
    def check_value(self, value):
        value = super(Str, self).check_value(value)
        try:
            return str(value)
        except TypeError:
            raise InvalidCfgAttr(f"{value} is not a valid string for attribute {self.name}")


class Bool(CfgAttr):
    def check_value(self, value):
        value = super(Bool, self).check_value(value)
        try:
            return bool(value)
        except TypeError:
            raise InvalidCfgAttr(f"{value} is not a valid boolean for attribute {self.name}")


class OneOf(CfgAttr):
    def __init__(self, *values, default='__undefined__'):
        self.values = []
        if values:
            for v in values:
                if isinstance(v, type):
                    self.values += [_type2attr(v)]
                else:
                    self.values += [v]
            super(OneOf, self).__init__(default)
        else:
            super(OneOf, self).__init__()
            self.values = None
            self.default = default

    def check_value(self, value):
        value = super(OneOf, self).check_value(value)
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


class StrMap(CfgAttr):
    def __init__(self, map: Mapping[str, any], default='__undefined__'):
        super(StrMap, self).__init__(default)
        self.map = map

    def check_value(self, value):
        try:
            value = super(StrMap, self).check_value(value)
            return self.map[value]
        except KeyError:
            raise InvalidCfgAttr(f"{value} is invalid for attribute {self.name} "
                                 f"(should be one of {list(self.map.keys())})")


class Obj(CfgAttr):
    def __init__(self, type=None, shortcut=None, default='__default__'):
        if type is not None:
            if default == '__default__':
                default = type()
            self.type = type
            if shortcut is not None and (not isinstance(shortcut, str) or shortcut not in type):
                    raise ValueError(f'Invalid shortcut name: {shortcut}. \n'
                                     f'(Valid shortcut are: {list(type.__attr__.keys())}).')
            self.shortcut = shortcut
            super(Obj, self).__init__(default)
        else:
            super(Obj, self).__init__()
            self.default = default
            self.shortcut = shortcut
            self.type = None

    def check_value(self, value):
        if not isinstance(value, Mapping):
            cfg = CfgDict.from_dict(value)
            value = self.type.from_cfg(cfg)
        elif not isinstance(value, self.type):
            obj = self.type()
            obj[self.shortcut] = value
            value = obj
        return super(Obj, self).check_value(value)


class Any(CfgAttr):
    pass


def _type2attr(type, default='__undefined__'):
    if issubclass(type, CfgAttr):
        return type(default=default)
    elif issubclass(type, CfgObj):
        return Obj(type=type, default=default)
    elif isinstance(type, Union):
        return OneOf(type.__args__, default=default)
    try:
        return {
            int: Int(default=default),
            float: Float(default=default),
            bool: Bool(default=default),
            str: Str(default=default),
        }[type]
    except KeyError:
        return Any(default=default)
