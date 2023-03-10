from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Dict, List, TypeVar, Generic, Iterable, Sized
import os
import weakref


UNDEFINED = '__undefined__'
UNSPECIFIED = '__unspecified__'
EMPTY = '__empty__'
HYPER_PARAMETER = '__hyper-parameter__'


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
    def __init__(self, cfg_dict, cfg_path: [str]):
        self._cfg_dict = cfg_dict
        self.cfg_path = cfg_path
        self.direction = 'continue'

    @property
    def parent(self) -> CfgDict:
        return self._cfg_dict[self.cfg_path[:-1]]

    @property
    def value(self):
        return self._cfg_dict[self.cfg_path]

    @value.setter
    def value(self, v):
        self._cfg_dict[self.cfg_path] = v

    def delete(self, remove_empty_roots=False):
        self._cfg_dict.delete(self.cfg_path, remove_empty_roots=remove_empty_roots)
        if remove_empty_roots:
            self.up()
        else:
            self.out()

    @property
    def mark(self):
        try:
            return self._cfg_dict.get_mark(self.cfg_path)
        except KeyError:
            return ""

    @property
    def name(self):
        return self.cfg_path[-1]

    @property
    def parent_fullname(self):
        return '.'.join(self.cfg_path[:-1])

    @property
    def fullname(self):
        return '.'.join(self.cfg_path)

    def out(self):
        self.direction = 'out'

    def up(self):
        self.direction = 'up'


class CfgDict(dict):
    @classmethod
    def from_dict(cls, data: Mapping[str, any]|Iterable[any], recursive=False, recursive_name=False, read_marks=False, **kwargs):
        """
        Cast a standard dictionary to a CfgDict.

        :param data: The dictionary to cast.
        :param recursive: If true any dictionary contained in data will be cast as well.
        :param recursive_name: If true name containing '.' will be interpreted as dictionaries of dictionaries.
            (Exemple: {'parent.child': 0} will be interpreted as {'parent': {'child': 0}}.)
        :param read_marks:
        :param kwargs:
        :return:
        """
        if data is None:
            return cls(**kwargs)
        elif isinstance(data, cls):
            return data
        elif isinstance(data, dict):
            from_dict_args = dict(recursive=recursive, recursive_name=recursive_name, read_marks=read_marks, **kwargs)

            r = cls(**kwargs)
            if isinstance(data, CfgDict):
                r.child_mark = data.child_mark
                r.mark = data.mark
                r._parent = data._parent

            for k, v in data.items():
                if read_marks:
                    if k == '__mark__':
                        r.mark = v
                        continue
                    elif k == '__child_marks__':
                        r.child_mark = v
                        continue
                if recursive_name:
                    if '.' in k:
                        k, child = k.split('.', 1)
                        v = CfgDict.from_dict({child: v}, **from_dict_args)
                if recursive:
                    if isinstance(v, list):
                        v = CfgDict.from_list(v, **from_dict_args)
                    elif is_dict(v):
                        if isinstance(v, CfgDict) and v.mark is not None:
                            r.child_mark[str(k)] = v.mark
                        v = CfgDict.from_dict(v, **from_dict_args)
                r[str(k)] = v
            return r
        elif isinstance(data, list):
            return cls.from_list(data, recursive=recursive, recursive_name=recursive_name, read_marks=read_marks, **kwargs)
        raise TypeError(f"Cannot cast {type(data)} to CfgDict.")

    @classmethod
    def from_list(cls, data: List[any], recursive=False, recursive_name=False, read_marks=False, allow_empty=True, **kwargs):
        if len(data) == 0:
            return cls(**kwargs)

        marks_set = {'__mark__', '__child_marks__'} if read_marks else set()
        if read_marks and isinstance(data[-1], dict) and '__mark__' in data[-1]:
            marks = data.pop(-1)
        else:
            marks = {}

        if not all((isinstance(_, dict) and len(set(_.keys())-marks_set) == 1) or
                   (allow_empty and isinstance(_, str)) for _ in data):
            return data

        dict_data = {m: deepcopy(marks[m]) for m in marks_set if m in marks}
        child_marks = dict_data.get('__child_marks__', None)
        for i, d in enumerate(data):
            if isinstance(d, dict):
                k = next(iter(d.keys()))
                v = next(iter(d.values()))
            else:
                k = d
                v = EMPTY
            dict_data[k] = v
            if child_marks and k not in child_marks and i in child_marks:
                child_marks[k] = child_marks.pop(i)

        return cls.from_dict(dict_data, recursive=recursive, recursive_name=recursive_name, read_marks=read_marks, **kwargs)

    def __init__(self, data: Dict[str, any] = None, parent=None, mark=None, child_mark=None):
        super(CfgDict, self).__init__()
        if isinstance(data, CfgDict):
            self.mark = data.mark
            self.child_mark = data.child_mark
            self._parent = data._parent if parent is None else weakref.ref(parent)
            self._name = data._name
        else:
            self.mark = mark
            self.child_mark = child_mark if child_mark is not None else {}
            self._parent = None if parent is None else weakref.ref(parent)
            self._name = None

        if data is not None:
            self.update(data)

    @property
    def parent(self):
        return None if self._parent is None else self._parent()

    @property
    def name(self)-> str | None:
        if self._name:
            return self._name
        parent = self.parent
        if parent is None:
            return None
        for k, v in parent.items():
            if v is self:
                self._name = k
                return k
        else:
            return None

    def roots(self, max_level=None):
        roots = []
        r = self
        i = 0
        while r.parent is not None and (max_level is None or i < max_level):
            r = r.parent
            roots += [r]
            i += 1
        return roots

    def root(self, max_level=None):
        roots = self.roots(max_level=max_level)
        return roots[-1] if roots else self

    def cfg_path(self) -> tuple[str] | tuple[()]:
        return tuple(_.name for _ in reversed(self.roots()[:-1]) if _.name)

    @property
    def fullname(self):
        if self._parent is None:
            return None
        path = self.cfg_path()
        if self.name:
            path = path + (self.name,)
        return '.'.join(path)

    def to_dict(self, flatten_path=False, exportable=False):
        if exportable:
            def format_v(v):
                from ..hyperparameters_tuning.generic_optimizer import HyperParameter
                match v:
                    case HyperParameter(): return v.full_specifications
                    case _: return v
        else:
            def format_v(v):
                return v

        if flatten_path:
            return {cursor.fullname: format_v(cursor.value) for cursor in self.walk_cursor()}
        else:
            return recursive_dict_map(self, lambda k, v: format_v(v))

    def __str__(self):
        s = ''
        if self.name:
            s = f'CfgDict[{self.name}]:\n'
        return s+self.to_yaml()

    def to_json(self):
        from json import dumps
        return dumps(self.to_dict(exportable=True))

    def to_yaml(self, file=None):
        import yaml
        return yaml.safe_dump(self.to_dict(exportable=True), stream=file, default_flow_style=False, sort_keys=False)

    def print_full_paths(self, prefix=''):
        return (prefix+'\n').join(f'{k}: {v}' for k, v in self.to_dict(flatten_path=True).items())

    def abs_path(self, rel_path, parent_node=None, root=None, check_exists=False):
        """
        Solve a relative path from this node. Return an existing parent node and a path to reach the given relative path
        from that node.
        :param rel_path: Path relative to this CfgDict
        :param parent_node: Choose the parent node from which the path will be expressed.
        If None (default), the returned node will be the closest existing parent node to the relative path destination.
        :param root: Set a node behond which the relative path may not look when moving upward in the hierarchy.
        :param check_exists: If true and the rel_path doesn't exist, an exception will be raised.
        :return: A tuple containing a CfgDict and a path to the relative path (as a tuple of strings).
        """
        if isinstance(rel_path, str):
            path = list(_.strip() for _ in rel_path.split('.'))
        elif isinstance(rel_path, tuple) and all(isinstance(_, str) for _ in rel_path):
            path = list(rel_path)
        else:
            raise TypeError('Invalid CfgDict key: %s.' % repr(rel_path))

        def simplify_path(path: list):
            i = len(path)-1
            to_remove = 0
            while i:
                if path[i] == '':
                    path.pop(i)
                    to_remove += 1
                elif to_remove:
                    to_remove -= 1
                    path.pop(i)
                i -= 1
            return ['']*to_remove + path


        if root is None:
            root = self.root()

        current_node = self
        while len(path) > 1:
            p = path.pop(0)
            if p == '':
                if current_node is root:
                    raise KeyError(f'Impossible to reach "{rel_path}" from "{self.fullname}":\n'
                                   f'Too many up in the hierarchy.')
                else:
                    current_node = current_node.parent
            else:
                n = current_node.get(p, None)
                if isinstance(n, CfgDict):
                    current_node = n
                else:
                    path.insert(0, p)
                    break

        if check_exists:
            r = current_node
            for i, p in enumerate(path):
                try:
                    r = r[p]
                except KeyError:
                    raise KeyError(f'Unknown path "{".".join(path[:i+1])}" from "{r.fullname}".') from None

        if parent_node is not None and parent_node is not current_node:
            roots = current_node.roots()
            try:
                id_parent = roots.index(parent_node) + 1
            except ValueError:
                raise ValueError(f'"{parent_node.fullname}" is not a parent of "{self.fullname}"') from None
            path = [_.name for _ in reversed(roots[:id_parent - 1])] + [current_node.name] + path
            current_node = parent_node

        return current_node, tuple(path)

    def get_mark(self, path):
        root, path = self.abs_path(path)

        if len(path) == 0:
            return root.mark
        return root[path[:-1]].child_mark.get(path[-1], None)

    def set_mark(self, path, mark):
        if mark is None:
            return

        m = self.get_mark(path)
        if m is not None:
            m.update(mark)
            return

        root, path = self.abs_path(path)
        if len(path) == 0:
            root.mark = mark
        else:
            root[path[:-1]].child_mark[path[-1]] = mark

    def __setitem__(self, key: str, value):
        if key == '__mark__':
            self.mark = value
            return
        elif key == '__child_marks__':
            self.child_mark = value
            return

        root, key = self.abs_path(key)

        if isinstance(value, dict):
            try:
                value = CfgDict.from_dict(value, recursive=True)
            except Exception:
                pass

        if len(key) == 1:
            if isinstance(value, CfgDict):
                value._parent = weakref.ref(root)
                value._name = key[0]
            if root is self:
                return super(CfgDict, self).__setitem__(key[0], value)
            else:
                root[key[0]] = value
                return

        r = root
        for i, k in enumerate(key[:-1]):
            if k not in r:
                r[k] = CfgDict()
            r = r[k]
        r[key[-1]] = value

    def __getitem__(self, item):
        if item == () or item == '':
            return self
        elif isinstance(item, tuple) and len(item) == 1 and isinstance(item[0], str):
            return super(CfgDict, self).__getitem__(item[0])
        elif isinstance(item, str):
            item = tuple(_.strip() for _ in item.split('.'))
            if len(item) == 1:
                return super(CfgDict, self).__getitem__(item[0])

        root, item = self.abs_path(item)

        r = root
        for i, it in enumerate(item):
            error = KeyError(f'Invalid item: "{".".join(item[:i+1])}".')
            if not isinstance(r, CfgDict):
                raise error
            try:
                r = r[it]
            except KeyError:
                raise error from None
        return r

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
        
    def __delitem__(self, item):
        root, item = self.abs_path(item)
        if len(item) == 1:
            item = item[0]
            if item in self.child_mark:
                del self.child_mark[item]
            node = self[item]
            if isinstance(node, CfgDict) and node.parent is self:
                node._parent = None
            return super(CfgDict, self).__delitem__(item)

        r = root[item[:-1]]
        del r[item[-1]]

    def __contains__(self, item):
        if isinstance(item, str) and '.' not in item:
            return super(CfgDict, self).__contains__(item)

        try:
            self.abs_path(item, check_exists=True)
        except KeyError:
            return False
        return True

    def pop(self, path, remove_empty_roots=False, with_mark=False):
        root, path = self.abs_path(path, check_exists=True)
        # r = reduce(lambda d, p: d[p], path[:-1], self)
        v = root[path[-1]]
        if with_mark:
            mark = v.child_mark.get(path[-1], None)
        del root[path[-1]]

        if remove_empty_roots:
            path = list(path)
            while path and len(r) == 0:
                name = path.pop()
                r = r.parent
                del r[name]

        return (v, mark) if with_mark else v

    def delete(self, path, remove_empty_roots=False):
        if isinstance(path, (list, tuple)):
            for p in path:
                try:
                    self.pop(p, remove_empty_roots=remove_empty_roots)
                except KeyError:
                    continue
        else:
            self.pop(path, remove_empty_roots=remove_empty_roots)

    def merge(self, __m: Mapping[str, any], **kwargs: any):
        d = self.copy()
        d.update(__m)
        d.update(kwargs)
        return d

    def update(self, __m: Dict[str, any], **kwargs: any) -> None:
        assert isinstance(__m, Mapping), f'Error when parsing {self.fullname}: __m must be a Mapping, not {type(__m)}'
        __m.update(kwargs)

        if isinstance(__m, CfgDict):
            self.child_mark.update(__m.child_mark)
            if __m.mark:
                if self.mark:
                    self.mark.update(__m.mark)
                else:
                    self.mark = __m.mark
        for k, v in __m.items():
            if isinstance(v, Mapping) and not isinstance(v, CfgDict):
                v = CfgDict.from_dict(v, recursive=True)
            elif isinstance(v, list):
                v = CfgDict.from_list(v, recursive=True)

            force_replace = k.endswith('!')
            if force_replace:
                if k in self.child_mark:
                    self.child_mark[k[:-1]] = self.child_mark.pop(k)
                k = k[:-1]

            if not force_replace and k in self and isinstance(self[k], CfgDict):
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

    def walk(self, only_leaf=False):
        for cursor in self._walk_cursor(only_leaf=only_leaf):
            yield cursor.fullname

    def walk_cursor(self, only_leaf=False) -> Iterable[CursorCfgDict]:
        for cursor in self._walk_cursor(only_leaf=only_leaf):
            yield cursor

    def _walk_cursor(self, rootpath=(), root_cfg=None, only_leaf=False) -> Iterable[CursorCfgDict]:
        if root_cfg is None:
            root_cfg = self
        for item in list(self.keys()):
            if item not in self:
                continue
            value = self[item]
            item_path = rootpath+(item,)
            if isinstance(value, CfgDict):
                for cursor in value._walk_cursor(rootpath=item_path, root_cfg=root_cfg):
                    if not only_leaf:
                        yield cursor
                    if cursor.direction != 'continue' and cursor.cfg_path[:-1] == rootpath:
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

    def current_versions(self) -> list[CfgVersion]:
        versions = []
        for c in self.walk_cursor(only_leaf=True):
            if c.mark:
                version = c.mark.version
                if version is not None and all(v is not versions for v in versions):
                    versions.append(version)
        return versions


T = TypeVar('T')


class CfgCollection(CfgDict):
    def __init__(self, obj_types, data=None, default_key=None, type_key=None, parent=None):
        self._default_key = default_key
        self.type_key = type_key

        if type_key is None:
            self._obj_types = [obj_types] if isinstance(obj_types, type) else obj_types
            assert isinstance(self.obj_types, (list, tuple)), (f"{type(self).__name__} "
                           f"Invalid obj_types: should be list when type_key is not provided.\n"
                           f"\tProvided value is:\n"+repr(self._obj_types))
            assert all(isinstance(_, type) for _ in self.obj_types)
        else:
            assert isinstance(obj_types, Mapping), f"{type(self).__name__} Invalid obj_types: should be dict when type_key is provided."
            assert all(isinstance(_, str) for _ in obj_types.keys())
            assert all(isinstance(_, type) for _ in obj_types.values())
            self._obj_types = obj_types

        super(CfgCollection, self).__init__(data, parent=parent)

    def __setitem__(self, key, value):
        try:
            mark = self.get_mark(key)
        except KeyError:
            mark = None

        value = self._to_obj_type(value, mark)
        return super(CfgCollection, self).__setitem__(key, value)

    def _init_after_populate(self):
        pass

    def init_after_populate(self):
        self._init_after_populate()
        for obj in self.values():
            from .cfg_object import CfgObj
            if getattr(obj, 'init_after_populate', None) is not None:
                obj.init_after_populate()

    @property
    def obj_types(self):
        return self._obj_types if self.type_key is None else list(self._obj_types.keys())

    def default(self):
        return None if self._default_key is None else self.get(self._default_key, None)

    def _to_obj_type(self, value, mark=None):
        if self.type_key is None:
            return self.to_type(self.obj_types, value, mark=mark)
        else:
            return self.match_type(self._obj_types, self.type_key, value, mark=mark)

    def match_type(self, obj_types: Mapping[str, type], type_key: str, value: any, mark=None):
        from .cfg_parser import ParseError
        t = value.get(type_key, None)
        if t is None:
            raise ParseError(f'Missing attribute {type_key} required to infer the object type', mark)
        obj_type = obj_types.get(t, None)
        if obj_type is None:
            raise ParseError(f'Invalid value "{t}" for attribute {type_key}. '
                             f'Should be one of {", ".join(obj_types.keys())}', mark)
        return self.to_type([obj_type], value, mark=mark)

    def to_type(self, obj_types: List[type], value: any, mark=None):
        from .cfg_parser import ParseError
        from .cfg_object import IncompleteObjError
        for obj_type in obj_types:
            if hasattr(obj_type, 'from_cfg') and isinstance(value, (dict, CfgDict)):
                try:
                    return obj_type.from_cfg(value, mark)
                except IncompleteObjError:
                    continue
            else:
                try:
                    return obj_type(value)
                except Exception:
                    continue

        if len(obj_types) == 1:
            raise ParseError(f"Item type {type(value).__name__} doesn't match Collection type {obj_types[0].__name__})",
                             mark)
        else:
            from .cfg_parser import format2str
            raise ParseError(f"Item type {type(value).__name__} doesn't match any Collection type "
                             f"{format2str([_.__name__ for _ in obj_types])}", mark)


class CfgCollectionRef:
    def __init__(self, key, collection):
        self.key = key
        self.collection = collection

    def ref(self):
        return self.collection[self.key]


class CfgList(CfgCollection, Generic[T]):
    def __init__(self, obj_types, shortcut_key: str, type_key: str=None, data=None, parent=None):
        self.shortcut_key = shortcut_key
        super(CfgList, self).__init__(obj_types=obj_types, type_key=type_key, data=data, parent=parent)

    def __iter__(self) -> Iterable[T]:
        return iter(self.values())

    def list(self) -> List[T]:
        return list(self)

    def __setitem__(self, key, value):
        if not isinstance(value, (dict, CfgDict)):
            value = CfgDict({self.shortcut_key: value if value is not EMPTY else key}, parent=self)
            value.mark = self.child_mark.get(key, None)
            if value.mark is not None:
                value.child_mark = {self.shortcut_key: value.mark}
        else:
            value = value.copy()
            value[self.shortcut_key] = key
            mark = self.child_mark.get(key, None)
            if mark is not None:
                value.child_mark[self.shortcut_key] = mark
        super(CfgList, self).__setitem__(key, value)


class CfgVersion:
    versions: list[CfgDict]
    version_id: int

    def __init__(self, version_id, versions):
        self.versions = versions
        self.version_id = version_id

    @property
    def current(self):
        return self.versions[self.version_id]


class Mark:
    def __init__(self, field_id: str, line: int, col: int, file, parser, version=None):
        from .cfg_parser import CfgFile

        self.field_id = field_id
        self.line = line
        self.col = col
        self.file = file if isinstance(file, CfgFile) else CfgFile(file, parser)
        self.version = version
        self._parser = weakref.ref(parser)

    def update(self, other_mark):
        self.field_id = other_mark.field_id
        self.line = other_mark.line
        self.col = other_mark.col
        self.file = other_mark.file
        self.version = other_mark.version
        self._parser = other_mark._parser

    def __str__(self):
        return f'in "{self.filename}", line {self.line}, column {self.col}'

    def __repr__(self):
        return f'Mark({self.line}, {self.col}, file="{self.filepath}")'

    @property
    def filename(self) -> str:
        return os.path.basename(self.file.path)

    @property
    def filepath(self) -> str:
        return self.file.path

    @property
    def fullpath(self) -> str:
        return os.path.abspath(self.file.path)

    @property
    def parser(self):
        return self._parser()

    def exception_like_description(self):
        exp_str = f'File "{self.fullpath}", line {self.line}'
        if self.field_id:
            return exp_str+f', in {self.field_id}'
        else:
            return exp_str
