import os
from copy import copy

import parse


class PathVar:
    def __init__(self, name, format=None):
        self.name = name
        if isinstance(format, str):
            parser = parse.compile('{:'+format+'}')
            def format_value(v):
                v = parser.parse(v)
                if v is None:
                    raise ValueError()
                return v[0]
            format = format_value

        self.format = format

    def __str__(self):
        return self.name

    @staticmethod
    def from_str(value: str):
        if ':' in value:
            name, format = value.split(':', 1)

            return PathVar(name.strip(), format.strip())
        else:
            return PathVar(value.strip())

    def parse(self, v):
        # TODO check format
        if self.format:
            v = self.format(v)
        return v


class PathToken(str):
    def __new__(cls, value):
        return super(PathToken, cls).__new__(cls, value)


class PathTemplate:
    def __init__(self, path_template: str, format_output='dict'):
        path = []
        self.vars = {}
        for i, p in enumerate(path_template.split('{')):
            if i == 0:
                path += [PathToken(p)]
            else:
                if '}' not in p:
                    raise ValueError('Invalid path template: a brace is never closed.')
                var, token = p.split('}', maxsplit=1)
                var = PathVar.from_str(var)
                if var.name in self.vars:
                    raise ValueError(f'Variable "{var.name}" is declared multiple time.')
                self.vars[var.name] = var
                path += [var]

                if token:
                    path += [PathToken(token)]
        self.path = path
        self.format_output = format_output

    def parse_filename(self, path: str):
        path = copy(path)
        template = copy(self.path)
        vars = {_: None for _ in self.vars.keys()}

        error = ValueError("filename doesn't match path template.")

        if isinstance(template[0], PathToken):
            if not path.startswith(template[0]):
                raise error
            temp = template.pop(0)
            path = path[len(temp):]

        if not template and not vars:
            return {}

        if isinstance(template[-1], PathToken):
            if not path.endswith(template[-1]):
                raise error
            temp = template.pop(-1)
            path = path[:-len(temp)]

        while path:
            temp = template.pop(-1)
            if isinstance(temp, PathToken):
                if not path.endswith(temp):
                    raise error
                path = path[-len(temp):]
            else:
                processed_vars = [temp]
                while template and isinstance(template[-1], PathVar):
                    processed_vars.append(template.pop(-1))

                if template:
                    path, path_vars = path.rsplit(template.pop(-1), maxsplit=1)
                else:
                    path_vars = path
                    path = ""

                for v in processed_vars:
                    path_vars, p = path_vars[:-1], path_vars[-1]
                    while path_vars:
                        try:
                            v.parse(p)
                        except ValueError:
                            break
                        else:
                            p = path_vars[-1] + p
                            path_vars = path_vars[:-1]
                    vars[v.name] = v.parse(p)
        return vars

    def parse_dir(self, dir, mode=None, recursive=False):
        vars = []

        if recursive:
            def listdir():
                for root, dirs, files in os.walk(dir):
                    if root.startswith(dir):
                        root = root[len(dir):]
                    for file in files:
                        yield os.path.join(root, file)
        else:
            def listdir():
                return os.listdir(dir)

        for f in listdir():
            match mode:
                case 'relative': path = os.path.join(dir, f)
                case 'basename': path = os.path.basename(f)
                case 'absolute': path = os.path.abspath(os.path.join(dir, f))
                case _: path = f

            try:
                var = self.parse_filename(path)
            except ValueError:
                continue
            else:
                var['fullpath'] = os.path.join(dir, f)
                vars.append(var)
        return self.post_process(vars)

    def post_process(self, vars):
        match self.format_output:
            case "dict":
                return vars
            case "pandas":
                import pandas as pd
                if vars:
                    if not self.vars:
                        return pd.DataFrame(vars)['fullpath']
                    else:
                        return pd.DataFrame(vars).set_index([_ for _ in vars[0].keys() if _ != 'fullpath'])['fullpath']
                else:
                    return pd.DataFrame()
