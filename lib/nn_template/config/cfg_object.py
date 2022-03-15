from collections.abc import Mapping
from types import Union

from .cfg_dict import CfgDict


class InvalidCfgAttr(Exception):
    def __init__(self, msg):
        super(InvalidCfgAttr, self).__init__(msg)


class CfgObj(CfgDict):
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
            if isinstance(attr_value, CfgDict):
                value = attr_value
            self._attr_values[attr_name] = attr_value
        super(CfgObj, self).__setitem__(key, value)

    @classmethod
    def from_cfg(cls, cfg_dict: CfgDict):
        r = cls()
        r.update(cfg_dict)
        return r


class MetaCfgAttr:
    def __new__(cls, *args, **kwargs):
        super(MetaCfgAttr, cls).__new__(*args, **kwargs)
        for attr_name, attr_type in cls.__annotations__.items():
            attr_value = getattr(cls, attr_name)
            if isinstance(attr_type, Union):
                pass
            else:
                for src_type, dest_type in _TYPED_ATTR.items():
                    if issubclass(attr_type, src_type):

                        break


    def __contains__(cls, item):
        return item in cls.__attr__


class CfgAttr(metaclass=MetaCfgAttr):
    def __init__(self, default_value='__undefined__'):
        self.name = ""
        if default_value is not None and default_value != '__undefined__':
            default_value = self.check_value(default_value)
        self.default = default_value

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

    def check_value(self, value):
        try:
            if isinstance(value, str):
                if value.endswith(tuple('TGMk')):
                    value = float(value[:-1])*{
                        'T': 1e12, 'G': 1e9, 'M': 1e6, 'k': 1e3
                    }[value[-1]]
            value = int(value)
        except TypeError:
            raise InvalidCfgAttr(f"{value} is not a valid integer for attribute {self.name}")
        if self.min is not None and self.min > value:
            raise InvalidCfgAttr(f"Provided value: {value}, exceed the minimum value {self.min} "
                                 f"for attribute {self.name}")
        if self.max is not None and self.max < value:
            raise InvalidCfgAttr(f"Provided value: {value}, exceed the maximum value {self.max} "
                                 f"for attribute {self.name}")
        return super(Int, self).check_value(value)


class Float(CfgAttr):
    def __init__(self, default='__undefined__', min=None, max=None):
        super(Float, self).__init__(default)
        self.min = min
        self.max = max

    def check_value(self, value):
        try:
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
            value = float(value)
        except TypeError:
            raise InvalidCfgAttr(f"{value} is not a valid float for attribute {self.name}")

        if self.min is not None and self.min > value:
            raise InvalidCfgAttr(f"Provided value: {value:.4e}, exceed the minimum value {self.min} "
                                 f"for attribute {self.name}")
        if self.max is not None and self.max < value:
            raise InvalidCfgAttr(f"Provided value: {value:.4e}, exceed the maximum value {self.max} "
                                 f"for attribute {self.name}")
        return super(Float, self).check_value(value)


class Str(CfgAttr):
    def check_value(self, value):
        try:
            return super(Str, self).check_value(str(value))
        except TypeError:
            raise InvalidCfgAttr(f"{value} is not a valid string for attribute {self.name}")


class Bool(CfgAttr):
    def check_value(self, value):
        try:
            return super(Bool, self).check_value(bool(value))
        except TypeError:
            raise InvalidCfgAttr(f"{value} is not a valid boolean for attribute {self.name}")


class OneOf(CfgAttr):
    def __init__(self, *values, default='__undefined__'):
        super(OneOf, self).__init__(default)
        self.values = values

    def check_value(self, value):
        for v in self.values:
            if isinstance(v, type):

        if value not in self.values:
            raise InvalidCfgAttr(f"{value} is invalid for attribute {self.name} (should be one of {self.values})")
        return super(OneOf, self).check_value(value)


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
    def __init__(self, type, shortcut=None, default='__default__'):
        if default == '__default__':
            default = type()
        self.type = type
        if shortcut is not None and (not isinstance(shortcut, str) or shortcut not in type):
                raise ValueError(f'Invalid shortcut name: {shortcut}. \n'
                                 f'(Valid shortcut are: {list(type.__attr__.keys())}).')
        self.shortcut = shortcut
        super(Obj, self).__init__(default)

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


_TYPED_ATTR = {
    int: Int,
    float: Float,
    bool: Bool,
    str: Str,
    CfgObj: Obj
}