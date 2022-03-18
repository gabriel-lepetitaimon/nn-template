import os
from itertools import product
from yaml import SafeLoader
from yaml.scanner import DirectiveToken
from yaml.parser import ParserError
import numpy as np

from .cfg_dict import CfgDict
from .cfg_object import CfgObj


_registered_cfg_object = {}


UNDEFINED = '__undefined__'


def register_obj(path: str, ):
    if path in _registered_cfg_object:
        raise ValueError(f'A configuration object is already registered at path "{path}"')

    def register(cfg_obj: CfgObj):
        _registered_cfg_object[path] = cfg_obj
        return cfg_obj
    return register


class ParseError(Exception):
    def __init__(self, error, mark=None):
        super(ParseError, self).__init__(error+(', '+str(mark)+'.' if mark is not None else '.'))
        self.mark = mark
        self.error = error


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
        return len(self.versions) if self.versions else 1

    def __getitem__(self, item):
        return self.get_config(item)

    def get_config(self, version=0, parse_obj=True):
        if self.base is None:
            self.parse()
        if version >= len(self):
            raise IndexError(f'Version index {version} out of range.')
        if not self.versions:
            cfg = self.base.copy()
        else:
            cfg = self.base.merge(self.versions[version])

        CfgParser.resolve_refs(cfg, inplace=True)
        if parse_obj:
            CfgParser.parse_registered_cfg(cfg, inplace=True)
        return cfg

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
            # Merge bases
            base.update(f.base)

            # Filter previous versions to remove fixed field
            inherited_versions = [filter_versions_with_base(v, base) for v in versions]
            versions = versions_product(*f.versions, inherited_versions=inherited_versions)

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
                v = cursor.value
                if isinstance(v, str) and v.startswith('$'):
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
                cfg_dict[path] = cfg_obj_class.from_cfg(cfg_dict[path])
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
        self.versions = versions + [[{k: _} for _ in l] for k, l in seq_versions.items()]
        self.base = base

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


def versions_product(*new_versions, inherited_versions=None):
    r = []
    new_versions = [_ for _ in new_versions if _]
    for idxs in product(*[range(len(_)) for _ in new_versions]):
        v = CfgDict()
        for i, version in reversed(list(zip(idxs, new_versions))):
            v.update(version[i])

        if inherited_versions:
            merged_versions = [inherited_version.merge(v)
                               for inherited_version in filter_versions_with_base(inherited_versions, v)]
            for merged_version in merged_versions:
                if merged_version not in r:
                    r.append(merged_version)
        elif v not in r:    # Check for duplicates
            r.append(v)
    return r


def filter_versions_with_base(versions, base):
    filtered_versions = []
    for version in versions:

        for cursor in version.walk_cursor():
            v = base.get(cursor.path, UNDEFINED)
            if v is not UNDEFINED:
                if v == cursor.value:
                    cursor.delete(remove_empty_roots=True)
                else:
                    break
        else:
            filtered_versions += [version]
    return filtered_versions


def simplify_versions(versions, base):
    if all(all(v in base for v in version.walk()) for version in versions):
        return versions, base

    # Remove versions which doesn't match base
    simplified_versions = []
    for version in versions:
        no_common_keys = True
        for cursor in version.walk_cursor():
            v = base.get(cursor.path, default=UNDEFINED)
            if v is not UNDEFINED:
                no_common_keys = False
                if cursor.value != base[cursor.path]:
                    break
                else:
                    cursor.delete(remove_empty_roots=True)
        else:
            if no_common_keys:
                continue
            simplified_versions += [version]
    versions = simplified_versions
    if len(versions) == 0:
        return [], base
    elif len(versions) == 1:
        return [], base.merge(versions[0])

    # Remove shared field with the same value (duplicates field)
    simplest_id = np.argmin(len(version) for version in versions)
    simplest_version = versions[simplest_id]
    other_versions = versions[:simplest_id]+versions[simplest_id+1:]
    for cursor in simplest_version:
        if all(cursor.value == version.get(cursor.path, default=UNDEFINED)
               for version in other_versions):
            base[cursor.path] = cursor.value
            cursor.delete(remove_empty_roots=True)
            for version in other_versions:
                version.delete(cursor.path, remove_empty_roots=True)

    return simplified_versions, base


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
            value = self.scan_inherit_value(start_mark)
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

    def scan_inherit_value(self, start_mark):
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
    def __init__(self, line:int, col:int, file:CfgFile|str):
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


