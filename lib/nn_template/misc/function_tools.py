__all__ = ['cached_not_null_property']


from types import GenericAlias
from time import time

_NOT_FOUND = object()


class cached_not_null_property:
    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it.")
        try:
            cache = instance.__dict__
        except AttributeError:  # not all objects have __dict__ (e.g. class defines slots)
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None
        val = cache.get(self.attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            val = self.func(instance)
            if val is not None:
                cache[self.attrname] = val
        return val

    __class_getitem__ = classmethod(GenericAlias)


class LogTimer:
    def __init__(self, process_name, log=True):
        self.process_name = process_name
        self.t0 = None
        self.log = log

    def __enter__(self):
        self.t0 = time()
        if self.log:
            print('  *** '+self.process_name+' ***')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self.log:
            print(f'     done in {time()-self.t0:.1f}s.')
