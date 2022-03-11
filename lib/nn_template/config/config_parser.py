from .config_dict import CfgDict
import os
from itertools import product
from yaml import SafeLoader
from yaml.scanner import DirectiveToken
from yaml.parser import ParserError


class CfgParser:
    def __init__(self, cfg_path):
        self.path = cfg_path
        self.files = []
        self.base = None
        self.versions = None

    def __len__(self):
        if self.base is None:
            return 0
        return len(self.versions) if self.versions else 1

    def __getitem__(self, item):
        return self.get_config(item)

    def get_config(self, version=0):
        if self.base is None:
            self.parse()
        if version >= len(self):
            raise IndexError(f'Version index {version} out of range.')
        if not self.versions:
            cfg = self.base.copy()
        else:
            cfg = self.base.merge(self.versions[version])
        return CfgParser.resolve_refs(cfg)

    def parse(self):
        self._parse()
        self._merge_files()

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
        self.base = self.files[-1].base
        for f in reversed(self.files[:-1]):
            self.base.update(f.base)
        versions = []
        for _ in self.files:
            if _.versions:
                versions += [_.versions]
        self.versions = versions_product(versions)

    @staticmethod
    def resolve_refs(cfg_dict: CfgDict):
        investigated = set()
        resolved = {}

        def resolve(path, rootpath):
            abs_path = cfg_dict.abs_path(path, rootpath, check_exists=True)
            if abs_path in resolved:
                return resolved[abs_path]
            if abs_path in investigated:
                raise RuntimeError(f'Redundent definition of "{".".join(abs_path)}" {cfg_dict.get_mark(abs_path)}.')
            investigated.add(abs_path)
            d = cfg_dict[abs_path]
            if isinstance(d, CfgDict):
                search_refs(abs_path)
            elif isinstance(d, str) and d.startswith('$'):
                try:
                    d = resolve(d[1:], abs_path[:-1])
                except KeyError:
                    raise ParserError(f'Unkown reference to "{d}", {cfg_dict.get_mark(abs_path)}.')
            resolved[abs_path] = d
            investigated.remove(abs_path)
            return d

        def search_refs(path):
            r = cfg_dict[path]
            for cursor in r.walk_cursor():
                v = cursor.value
                if isinstance(v, str) and v.startswith('$'):
                    try:
                        cursor.value = resolve(v[1:], '.'.join((path+cursor.path)[:-1]))
                    except KeyError:
                        raise ParserError(f'Unkown reference to "{v}", {cursor.mark}.')
        search_refs(())
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
        self.versions = versions_product([[{k: _} for _ in l] for k, l in seq_versions.items()] + [versions])
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
                r[cursor.name[:-1]] = cursor.value[0]
                del r[cursor.name]
                cursor.out()
        return seq_v


def versions_product(versions):
    r = []
    for idxs in product(*[range(len(_)) for _ in versions]):
        v = CfgDict()
        for i, version in reversed(list(zip(idxs, versions))):
            v.update(version[i])
        r.append(v)
    return r


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