from functools import reduce
from typing import Mapping, List
from collections.abc import Iterable


def is_dict(o):
    return isinstance(o, (dict, CfgDict))


def recursive_dict_update(destination, origin, append: str | bool = False):
    for k, v in origin.items():
        dest_v = destination.get(k, None)
        if is_dict(v) and is_dict(dest_v):
            recursive_dict_update(destination[k], v)
        elif append and isinstance(v, list) and isinstance(dest_v, list):
            for list_v in v:
                append_needed = True
                if is_dict(list_v) and isinstance(append, str) and append in list_v:
                    key = list_v[append]
                    for i in range(len(dest_v)):
                        if is_dict(dest_v[i]) and dest_v[i] and dest_v[i].get(append, None) == key:
                            recursive_dict_update(dest_v[i], list_v, append=append)
                            append_needed = False
                if append_needed:
                    dest_v.append(list_v)
        else:
            destination[k] = v


def recursive_dict_map(dictionnary, function):
    r = {}
    for n, v in dictionnary.items():
        if is_dict(v):
            v = recursive_dict_map(v, function=function)
        else:
            v = function(n, v)
        if v is not None:
            r[n] = v
    return r


class CursorCfgDict:
    def __init__(self, cfg_dict, path: [str]):
        self._cfg_dict = cfg_dict
        self.path = path
        self.direction = 'continue'

    @property
    def parent(self):
        return self._cfg_dict[self.path[:-1]]

    @property
    def value(self):
        return self._cfg_dict[self.path]

    @value.setter
    def value(self, v):
        self._cfg_dict[self.path] = v

    @property
    def mark(self):
        try:
            return self._cfg_dict.get_mark(self.path)
        except KeyError:
            return ""

    @property
    def name(self):
        return self.path[-1]

    @property
    def parent_fullname(self):
        return '.'.join(self.path[:-1])

    @property
    def fullname(self):
        return '.'.join(self.path)

    def out(self):
        self.direction = 'out'

    def up(self):
        self.direction = 'up'


class CfgDict(dict):
    @staticmethod
    def from_dict(d, recursive=False, read_marks=False, parent=None):
        if isinstance(d, list) and all(isinstance(_, dict) and len(_)==1 for _ in d):
            l = d
            d = {}
            for _ in l:
                k0 = next(iter(_.keys()))
                d[k0] = _[k0]
        if isinstance(d, CfgDict):
            return d
        elif isinstance(d, dict):
            r = CfgDict(parent=parent)
            for k, v in d.items():
                if read_marks:
                    if k == '__mark__':
                        r.mark = v
                        continue
                    elif k =='__child_marks__':
                        r.child_mark = v
                        continue
                if is_dict(v) and recursive:
                    if isinstance(v, CfgDict) and v.mark is not None:
                        r.child_mark[str(k)] = v.mark
                    v = CfgDict.from_dict(v, recursive=True, read_marks=read_marks, parent=r)
                r[str(k)] = v
            return r
        return d

    def __init__(self, parent=None):
        super(CfgDict, self).__init__()
        self.mark = None
        self.child_mark = {}

        import weakref
        self._parent = None if parent is None else weakref.ref(parent)
        self._name = None

    @property
    def parent(self):
        return None if self._parent is None else self._parent()

    @property
    def name(self):
        if self._name:
            return self._name
        parent = self.parent
        if parent is None:
            return None
        for k, v in self.parent.items():
            if v is self:
                self._name = k
                return k
        else:
            return None

    def roots(self, max_level=None):
        roots = []
        r = self
        i=0
        while r.parent is not None and (max_level is None or i<max_level):
            r = r.parent
            roots += [r]
            i += 1
        return roots

    def root(self, max_level=None):
        roots = self.roots(max_level=max_level)
        return roots[-1] if roots else self

    def path(self):
        return tuple(_.name for _ in reversed(self.roots()))

    @property
    def full_name(self):
        return None if self._parent is None else '.'.join(self.path()+(self.name,))

    def to_dict(self):
        return recursive_dict_map(self, lambda k, v: v)

    def __str__(self):
        s = ''
        if self.name:
            s = f'CfgDict[{self.name}]:\n'
        return s+self.to_yaml()

    def __repr__(self):
        return 'CfgDict()'

    def to_json(self):
        from json import dumps
        return dumps(self)

    def to_yaml(self, file=None):
        import yaml
        return yaml.dump(self.to_dict(), stream=file, default_flow_style=False)

    def abs_path(self, rel_path, root=None, check_exists=False):
        path = tuple(_.strip() for _ in rel_path.split('.'))
        if root is None:
            root = self.root()
        if root is not self:
            roots = self.roots()
            try:
                id_root = roots.index(root)
            except ValueError:
                raise ValueError(f'"{root.full_name}" is not a parent of "{self.full_name}"') from None
            path = tuple(reversed(roots[:id_root]))+(self.name,)+path

        abs_path = []
        for p in path:
            if p:
                abs_path += [p]
            elif len(abs_path):
                abs_path.pop()
            else:
                raise KeyError(f'Impossible to reach "{rel_path}" from "{self.full_name}":\n'
                                 f'Too many up in the hierarchy.')
        abs_path = tuple(abs_path)

        if check_exists:
            r = root
            for i, p in enumerate(abs_path):
                try:
                    r = r[p]
                except KeyError:
                    raise KeyError(f'Unknown path "{".".join(abs_path[:i+1])}".') from None
        return abs_path

    def get_mark(self, path):
        if isinstance(path, str):
            path = self.abs_path(path)
        if len(path) == 0:
            return self.mark
        return self['.'.join(path[:-1])].child_mark[path[-1]]

    def __setitem__(self, key: str, value):
        if isinstance(key, str):
            key = self.abs_path(key)
        elif not isinstance(key, tuple) or any(not isinstance(_, str) for _ in key):
            raise TypeError('Invalid CfgDict key: %s.' % repr(key))
        if isinstance(value, dict):
            try:
                value = CfgDict.from_dict(value, recursive=True, parent=self)
            except:
                pass
        if len(key) == 1:
            return super(CfgDict, self).__setitem__(key[0], value)

        r = self
        for i, k in enumerate(key[:-1]):
            if k not in r:
                r[k] = CfgDict()
            r = r[k]
        r[key[-1]] = value

    def __getitem__(self, item):
        if item == () or item == '':
            return self
        elif isinstance(item, str):
            item = self.abs_path(item)
        elif not isinstance(item, tuple) or any(not isinstance(_, str) for _ in item):
            raise TypeError('Invalid CfgDict key: %s.' % repr(item))
        if len(item) == 1:
            return super(CfgDict, self).__getitem__(item[0])

        r = self
        for i, it in enumerate(item):
            try:
                r = r[it]
            except KeyError:
                raise KeyError(f'Invalid item: {".".join(item[:i])}.') from None
        return r
        
    def __delitem__(self, item):
        if isinstance(item, str):
            item = self.abs_path(item)
        elif not isinstance(self, tuple) or any(not isinstance(_, str) for _ in item):
            raise TypeError('Invalid CfgDict key: %s.' % repr(item))
        if len(item) == 1:
            item = item[0]
            if item in self.child_mark:
                del self.child_mark[item]
            return super(CfgDict, self).__delitem__(item)

        r = self[item[:-1]]
        del r[item[-1]]

    def __contains__(self, item):
        if not isinstance(item, str):
            raise TypeError('Invalid CfgDict key: %s.' % repr(item))
        if '.' not in item:
            return super(CfgDict, self).__contains__(item)

        try:
            self.abs_path(item, check_exists=True)
        except KeyError:
            return False
        return True

    def pop(self, path):
        path = self.abs_path(path)
        r = reduce(lambda r, p: r[p], path[:-1], self)
        v = r[path[-1]]
        del r[path[-1]]
        return v

    def merge(self, __m: Mapping[str, any], **kwargs: any):
        d = self.copy()
        d.update(__m)
        return d

    def update(self, __m: Mapping[str, any], **kwargs: any) -> None:
        __m.update(kwargs)

        if isinstance(__m, CfgDict):
            self.child_mark.update(__m.child_mark)
        for k, v in __m.items():
            if is_dict(v) and not isinstance(v, CfgDict):
                v = CfgDict.from_dict(v, recursive=True, parent=self)
            if k in self and isinstance(self[k], CfgDict):
                self[k].update(v)
            else:
                self[k] = v

    def filter(self, condition, recursive=False):
        for k in list(self.keys()):
            v = self[k]
            if recursive and isinstance(v, CfgDict):
                v.filter(condition, True)
            elif not condition(k, v):
                del self[k]
        return self

    def map(self, f, recursive=False, remove_if_none=False):
        for k, v in self.items():
            if recursive and isinstance(v, CfgDict):
                v.map(f, True, remove_if_none=remove_if_none)
            else:
                r = f(k, v)
                if remove_if_none and r is None:
                    del self[k]
                else:
                    self[k] = r
        return self

    def walk(self):
        for root, k, v in self.walk_cursor():
            yield f"{root}.{k}"

    def walk_cursor(self, rootpath=(), root_cfg=None) -> Iterable[CursorCfgDict]:
        if root_cfg is None:
            root_cfg = self
        for item in list(self.keys()):
            if item not in self:
                continue
            value = self[item]
            item_path = rootpath+(item,)
            if isinstance(value, CfgDict):
                for cursor in value.walk_cursor(rootpath=item_path, root_cfg=root_cfg):
                    yield cursor
                    if cursor.direction != 'continue' and cursor.path[:-1] == rootpath:
                        if cursor.direction == 'up':
                            return
                        elif cursor.direction == 'out':
                            break
            else:
                yield CursorCfgDict(root_cfg, item_path)

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def subset(self, items):
        from copy import deepcopy
        r = CfgDict()
        if isinstance(items, str):
            items = items.split(',')
            items = [_.strip() for _ in items]
        for it in items:
            r[it] = deepcopy(self[it])
        return r

    def check(self, path, value=True, missing=None):
        path = path.split('.')

        miss = False
        item = self
        for i, p in enumerate(path):
            if p not in item:
                miss = True
                path = '.'.join(path[:i])
                break
            item = item[p]

        if miss:
            if missing is None:
                raise AttributeError(f'Missing attribute {path}.')
            return missing

        if isinstance(value, bool) or value is None:
            return item is value
        if isinstance(value, tuple) and not isinstance(item, tuple):
            return item in value
        return item == value
