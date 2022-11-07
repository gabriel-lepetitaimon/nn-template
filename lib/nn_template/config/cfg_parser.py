import os
from yaml import SafeLoader
from yaml.scanner import DirectiveToken
from yaml.parser import ParserError
from collections import abc
import numpy as np

from .cfg_dict import CfgDict
from .cfg_object import UNDEFINED, CfgCollectionType


_registered_cfg_object = {}


def register_obj(path: str, collection=False, type=None):
    if path not in _registered_cfg_object:
        _registered_cfg_object[path] = {} if collection else None
    else:
        registered_obj = _registered_cfg_object[path]

        from .cfg_object import InvalidAttrError
        if isinstance(registered_obj, CfgCollectionType) != bool(collection):
            if collection:
                raise InvalidAttrError(f'Configuration objects was already registered at path="{path}" as a non-collection.')
            else:
                raise InvalidAttrError(f'Configuration objects was already registered at path="{path}" as a collection.')
        elif not collection and type in path:
            raise InvalidAttrError(f'A configuration object is already registered at path="{path}", type="{type}".')

    def register(cfg_obj: type):
        if not collection:
            _registered_cfg_object[path].update({type:  cfg_obj})
        else:
            default_key = collection if isinstance(collection, str) else None
            cfg_collection: CfgCollectionType = _registered_cfg_object.get(path, None)
            if cfg_collection is None:
                _registered_cfg_object[path] = CfgCollectionType(obj_types=cfg_obj, default_key=default_key)
            else:
                if cfg_obj in cfg_collection.obj_types:
                    raise InvalidAttrError(f'A configuration object is already registered at path="{path}", type="{cfg_obj}".')
                cfg_collection.obj_types = cfg_collection.obj_types + (cfg_obj,)
        return cfg_obj
    return register


class ParseError(Exception):
    def __init__(self, error, mark=None, info=None):
        if error.endswith('.'):
            error = error[:-1]
        super(ParseError, self).__init__(error + (', '+str(mark)+'.' if mark is not None else '.')
                                         + ('\n\t' + info.replace('\n', '\n\t') if info else ''))
        self.mark = mark
        self.error = error
        self.info = info


def format2str(v):
    match v:
        case str():
            v.replace('"', '\\"')
            return f'"{v}"'
        case float():
            return f"{v:.4e}"
        case abc.Iterable():
            return ', '.join(format2str(_) for _ in v)
        case _:
            return str(v)


class CfgParser:
    def __init__(self, cfg_path):
        self.path = cfg_path
        self.files = []
        self.base = None
        self.versions = None

    Error = ParseError

    def __len__(self):
        if self.base is None:
            return 0
        return np.prod(len(v) for v in self.versions) if self.versions else 1

    def __getitem__(self, item):
        return self.get_config(item)

    def get_config(self, version=0, parse_obj=True):
        if self.base is None:
            self.parse()
        if not self.versions:
            cfg = self.base.copy()
        else:
            cfg = self.base.merge(self.get_version(version))

        CfgParser.resolve_refs(cfg, inplace=True)
        if parse_obj:
            try:
                CfgParser.parse_registered_cfg(cfg, inplace=True)
            except ParseError as e:
                raise ParseError(error=e.error, mark=e.mark, info=e.info) from None
        return cfg

    def get_version(self, i):
        shapes = tuple(len(v) for v in self.versions)
        if i >= np.prod(shapes):
            raise IndexError(f'Version index {i} out of range.')
        idxs = np.unravel_index(i, shapes)
        v = CfgDict()
        for i, versions in zip(idxs, self.versions):
            v.update(versions[i])
        return v

    def parse(self):
        self._parse()
        self._merge_files()
        return self

    def _parse(self):
        f = CfgFile(self.path).parse()
        self.files = [f]
        dependencies = list(f.inherit)
        while dependencies:
            f = dependencies.pop(0).parse()
            self.files += [f]
            for d in f.inherit:
                if d not in self.files and d not in dependencies:
                    dependencies += [d]

    def _merge_files(self):
        base = CfgDict()
        versions = []

        for f in reversed(self.files):
            versions, base = merge_versions_bases(versions, base, f.versions, f.base)

        self.versions = versions
        self.base = base

    @staticmethod
    def resolve_refs(cfg_dict: CfgDict, inplace=False):
        if not inplace:
            cfg_dict = cfg_dict.copy()

        investigated = set()
        resolved = {}

        def resolve(ref, parent_node):
            ref = ref.strip()
            if ref.startswith('.'):     # Relative ref
                ref = ref[1:]
            else:                       # Absolute ref
                parent_node = cfg_dict

            rel_root, rel_path = parent_node.abs_path(ref)
            abs_path = rel_root.path() + rel_path
            if abs_path in resolved:
                return resolved[abs_path]
            if abs_path in investigated:
                raise ParseError(f'Redundent definition of "{".".join(abs_path)}"', rel_root.get_mark(rel_path))
            investigated.add(abs_path)

            node = rel_root
            for p in rel_path:
                if not isinstance(node, CfgDict):
                    raise KeyError
                parent = node
                node = parent[p]
                if isinstance(node, str) and node.startswith('$'):
                    try:
                        node = resolve(node[1:], parent)
                    except KeyError:
                        raise ParseError(f'Unknown reference to "{node}"', parent.get_mark(p)) from None
                    parent[p] = node

            if isinstance(node, CfgDict):
                search_resolve_refs(node)
            resolved[abs_path] = node
            investigated.remove(abs_path)
            return node

        def search_resolve_refs(node):
            for cursor in node.walk_cursor():
                k = cursor.name
                v = cursor.value
                if k == "$":
                    try:
                        v = resolve(v, cursor.parent)
                    except KeyError:
                        raise ParseError(f'Unknown reference to "{v}"', cursor.mark) from None
                    if not isinstance(v, CfgDict):
                        raise ParseError(f'Invalid reference to populate {cursor.name}: '
                                         f'"{v}" is not a dictionary', cursor.mark) from None
                    v = v.merge(cursor.parent)
                    cursor.parent.update(v)
                    del cursor.parent['$']
                elif isinstance(v, str) and v.startswith('$'):
                    try:
                        cursor.value = resolve(v[1:], cursor.parent)
                    except KeyError:
                        raise ParseError(f'Unknown reference to "{v}"', cursor.mark) from None

        try:
            search_resolve_refs(cfg_dict)
        except ParseError as e:
            raise ParseError(e.error, e.mark) from None
        return cfg_dict

    @staticmethod
    def parse_registered_cfg(cfg_dict: CfgDict, inplace=False):
        if not inplace:
            cfg_dict = cfg_dict.copy()

        for path, cfg_obj_class in _registered_cfg_object.items():
            if path in cfg_dict:
                if isinstance(cfg_obj_class, dict):
                    if len(cfg_obj_class) == 1 and None in cfg_obj_class:
                        cfg_obj_class = cfg_obj_class[None]
                    else:
                        type = cfg_dict.get('type', UNDEFINED)
                        if type is UNDEFINED:
                            raise ParseError(f'Missing a "type" attribute for {path}', cfg_dict.get_mark(path))
                        if type not in cfg_obj_class:
                            raise ParseError(f'Invalid type attribute for {path}', cfg_dict[path].get_mark('type'),
                                             f"Should be one of: {format2str(cfg_obj_class.keys())}.")
                        cfg_obj_class = cfg_obj_class[type]
                cfg_dict[path] = cfg_obj_class.from_cfg(cfg_dict[path], mark=cfg_dict.get_mark(path), path=path)

        for path in _registered_cfg_object.keys():
            if path in cfg_dict:
                from .cfg_object import ObjCfg, CfgCollection
                obj = cfg_dict[path]
                if isinstance(obj, ObjCfg):
                    obj.check_integrity(True)
                elif isinstance(obj, CfgCollection):
                    for item in obj.values():
                        if isinstance(item, ObjCfg):
                            item.check_integrity(True)

        for path in _registered_cfg_object.keys():
            cfg_dict[path].init_after_populate()

        return cfg_dict


class CfgFile:
    """
    Represent and parse an actual yaml of json file containing configuration information.
    Apart from the file base data, it stores all the versions of this file and links to inherited CfgFiles.
    """
    def __init__(self, path: str):
        self.path = path
        self.base = None
        self.inherit = None
        self.versions = None

    def __repr__(self):
        return f"CfgFile(path={self.path})"

    def __eq__(self, other):
        return isinstance(other, CfgFile) and other.path == self.path

    def parse(self):
        if self.path.endswith('.yaml'):
            yaml_docs = []
            with open(self.path, 'r') as yaml_file:
                loader = CfgYamlLoader(yaml_file, self)
                try:
                    while loader.check_data():
                        yaml_docs += [loader.get_data()]
                finally:
                    loader.dispose()
            base = CfgDict.from_dict(yaml_docs[0], recursive=True, read_marks=True)
            versions = [CfgDict.from_dict(_, recursive=True, read_marks=True) for _ in yaml_docs[1:]]
            dirname = os.path.dirname(self.path)
            self.inherit = [CfgFile(os.path.join(dirname, _)) for _ in loader.inherit]

        elif self.path.endswith('.json'):
            import json
            base = json.loads(self.path)
            base = CfgDict.from_dict(base, recursive=True)
            versions = []
            self.inherit = []
        else:
            raise ValueError(f'Invalid configuration extension: "{os.path.basename(self.path)}". '
                             f'(Only yaml and json are accepted.)')

        seq_versions = CfgFile.parse_sequence_versions(base)
        for v in versions:
            for c in v.walk_cursor():
                if c.path in seq_versions.keys() or c.path in base:
                    raise ParseError(f'Attribute {c.path} is already defined', c.mark)
        versions, self.base = curate_versions_base(versions, base)

        self.versions = [[CfgDict({k: _}) for _ in l] for k, l in seq_versions.items()]
        if versions:
            self.versions.append(versions)

        return self

    @staticmethod
    def parse_sequence_versions(cfg_dict: CfgDict):
        seq_v = {}
        for cursor in cfg_dict.walk_cursor():
            if cursor.name.endswith('@') and isinstance(cursor.value, list) and len(cursor.value):
                seq_v[cursor.fullname[:-1]] = cursor.value
                r = cfg_dict[cursor.parent_fullname]
                r.child_mark[cursor.name[:-1]] = r.child_mark[cursor.name]
                del r[cursor.name]
                cursor.out()
        return seq_v


def curate_versions_base(versions, base):
    """
    Remove versions that collide with the base (same keys different values).
    Fill fields present in some versions but missing from other with its value in base.
    Remove duplicated versions.
    If only one version match the base, it is merged into the base and an empty versions list is returned.
    If a field is the same in all versions, it is moved to the base.

    Returns: Curated versions and base (base and versions have no fields in common, versions have no duplicates)
    """
    versions_keys = {_ for version in versions for _ in version.walk()}
    shared_keys = {}
    for k in versions_keys:
        try:
            shared_keys[k] = base[k]
        except KeyError:
            continue
    if not shared_keys:
        return versions, base

    # Remove versions which doesn't match base
    simplified_versions = []
    for version in versions:
        for k, base_value in shared_keys.items():
            try:
                version_value = version[k]
            except KeyError:
                version[k] = base_value     # If missing, fill with base_value
            else:
                if base_value != version_value:
                    break
        else:   # If version match
            if version not in simplified_versions:  # and is not a duplicate
                simplified_versions.append(version)

    versions = simplified_versions
    if len(versions) == 0:
        return [], base
    elif len(versions) == 1:
        return [], base.merge(versions[0])

    # Remove shared field with the same value (duplicates field)
    simplest_id = np.argmin(len(version) for version in versions)
    simplest_version = versions[simplest_id]
    other_versions = versions[:simplest_id]+versions[simplest_id+1:]
    for cursor in simplest_version.walk_cursor():
        if all(cursor.value == version.get(cursor.path, default=UNDEFINED)
               for version in other_versions):
            base[cursor.path] = cursor.value
            cursor.delete(remove_empty_roots=True)
            for version in other_versions:
                version.delete(cursor.path, remove_empty_roots=True)

    return simplified_versions, base


def merge_versions_bases(inherited_versions, inherited_base, new_versions, new_base):
    """
    Assume that inherited_versions and inherited_base are curated.
    Assume that new_versions and new_base are curated.

    Returns:
    """

    # --- Remove fields present in new_versions from inherited_base ---
    new_versions_keys = []
    for versions in new_versions:
        keys = {k for v in versions for k in v.walk()}

        for k in keys:
            try:
                # Remove new_versions fields from inherited_base
                inh_base_v = inherited_base.pop(k, remove_empty_roots=True)
            except KeyError:
                continue
            for version in versions:
                if k not in version:
                    version[k] = inh_base_v     # Fill new_versions with inherited_base value
        new_versions_keys.append(keys)

    # --- Curate inherited_versions and new_base
    for i, versions in enumerate(inherited_versions):
        inherited_versions[i], new_base = curate_versions_base(versions, new_base)
    inherited_versions = [v for v in inherited_versions if v]

    # --- Curate inherited_versions and new_versions ---
    curated_inh_versions = []
    for i, inh_versions in enumerate(inherited_versions):
        already_fused = False
        inh_keys = {k for v in inh_versions for k in v.walk()}
        for keys, (n, versions) in zip(new_versions_keys, enumerate(new_versions)):
            shared_keys = inh_keys.union(keys)
            if shared_keys:
                if not already_fused:
                    fused_versions = []
                    for version in versions:
                        fused_v, version = curate_versions_base(inh_versions, version)
                        if fused_v:
                            fused_versions += [version.merge(v) for v in fused_v]
                        else:
                            fused_versions.append(version)
                    new_versions[n] = fused_versions
                    already_fused = shared_keys, versions
                else:
                    # conflict_keys = shared_keys.union(already_fused)
                    raise ParseError(f'Conflict when resolving inherited versions.')
        if not already_fused:
            curated_inh_versions.append(inh_versions)
    versions = curated_inh_versions + new_versions

    base = inherited_base.merge(new_base)
    return versions, base


class CfgYamlLoader(SafeLoader):
    def __init__(self, stream: any, file: CfgFile):
        super(CfgYamlLoader, self).__init__(stream=stream)
        self.file = file
        self.inherit = []

    def construct_mapping(self, node, deep=False):
        mapping = super(CfgYamlLoader, self).construct_mapping(node, deep=deep)
        # Add 1 so line numbering starts at 1
        mark = node.start_mark
        mapping["__mark__"] = Mark(mark.line, mark.column-1, self.file)
        child_marks = {}
        for k, v in node.value:
            mark = v.start_mark
            child_marks[k.value] = Mark(mark.line+1, mark.column, self.file)
        mapping["__child_marks__"] = child_marks
        return mapping

    def scan_directive(self):
        # -- Code copied from yaml.scanner.scan_directive --
        # See the specification for details.
        start_mark = self.get_mark()
        self.forward()
        name = self.scan_directive_name(start_mark)
        value = None
        if name == 'YAML':
            value = self.scan_yaml_directive_value(start_mark)
            end_mark = self.get_mark()
        elif name == 'TAG':
            value = self.scan_tag_directive_value(start_mark)
            end_mark = self.get_mark()
        elif name == "INHERIT":
            value = self.scan_inherit_value()
            if value in self.inherit:
                raise ParserError("while parsing inheritance", self.marks[-1],
                                  f'"{value}" is already inherited.', start_mark)
            self.inherit += [value]
            end_mark = self.get_mark()
        else:
            end_mark = self.get_mark()
            while self.peek() not in '\0\r\n\x85\u2028\u2029':
                self.forward()
        self.scan_directive_ignored_line(start_mark)
        return DirectiveToken(name, value, start_mark, end_mark)

    def scan_inherit_value(self):
        while self.peek() == ' ':
            self.forward()
        value = ""
        ch = self.peek()
        while ch not in '\0\r\n\x85\u2028\u2029':
            value += ch
            self.forward()
            ch = self.peek()
        return value


class Mark:
    def __init__(self, line: int, col: int, file: CfgFile | str):
        self.line = line
        self.col = col
        self.file = file if isinstance(file, CfgFile) else CfgFile(file)

    def __str__(self):
        return f'in "{self.filename}", line {self.line}, column {self.col}'

    def __repr__(self):
        return f'Mark({self.line}, {self.col}, file="{self.filepath}")'

    @property
    def filename(self):
        return os.path.basename(self.file.path)

    @property
    def filepath(self):
        return self.file.path
