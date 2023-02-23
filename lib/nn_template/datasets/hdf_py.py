import numpy as np
import h5py
import os.path as P
from copy import deepcopy
from torch.utils.data import Dataset
from typing import Dict, List

from ..config import Cfg
from ..data_augmentation import DataAugment


def create_generic_hdf_datasets(dataset_cfg: Cfg.Dict, file_path=None, seed=1234,
                                data_augmentations: Dict[str,DataAugment]=None):
    from .datasets import DEFAULT_DATA_PATH
    if file_path is None:
        file_path = DEFAULT_DATA_PATH
    hdf_path = P.join(file_path, dataset_cfg.cfg_path)

    train = create_generic_hdf_dataset(dataset_cfg, 'task', hdf_path, seed, data_augmentations)
    valid = create_generic_hdf_dataset(dataset_cfg, 'validation', hdf_path, seed, data_augmentations)
    test = create_generic_hdf_dataset(dataset_cfg, 'testing', hdf_path, seed, data_augmentations)
    return train, valid, test


def create_generic_hdf_dataset(datasets_cfg: Cfg.Dict, prefix: str, hdf_path: str,
                               seed, data_augmentations: Dict[str, DataAugment]):
    cfg = Cfg.Dict.from_dict({
        'dataset': 'all',
        'augment': False,
        'preload-in-RAM': prefix == 'validation',
    })
    if prefix in datasets_cfg:
        cfg.update(datasets_cfg[prefix])
    datasets = cfg.dataset

    fields_mapping = datasets_cfg.fields
    hdf_vatiables = set()
    for m in fields_mapping.values():
        hdf_vatiables.update(extract_variable_from_expr(m))

    if cfg['augment'] is False:
        data_augmentations = None
        factor = 1
        augmented_fields = None
    elif isinstance(cfg.augment, (Cfg.Dict, dict)):
        unkown_fields = {k for k in cfg.augment.fields.keys() if k not in fields_mapping}
        if unkown_fields:
            raise ValueError(f'Unkown data field(s) {", ".join(unkown_fields)} in data augmentation specs of '
                             f'the {prefix} dataset using file {hdf_path}.')
        augmentation_names = cfg.augment.get('data-augmentations', 'default')
        if isinstance(augmentation_names, str):
            augmentation_names = [augmentation_names]
        unkown_data_augmentation = {a for a in augmentation_names if a not in data_augmentations}
        if unkown_data_augmentation:
            raise ValueError(f'Unkown data field(s) {", ".join(unkown_data_augmentation)} in data augmentation specs of '
                             f'the {prefix} dataset using file {hdf_path}.')
        data_augmentations = [data_augmentations[a] for a in augmentation_names]
        factor = cfg.augment.get('factor', 1)
        augmented_fields = cfg.augment.fields
    else:
        raise ValueError(f'Invalid augment value for dataset {prefix} using file {hdf_path}.')

    ## For backward compatibility.
    PREFIX_FALLBACKS = {'task': 'train', 'validation': 'val', 'testing': 'test'}

    ## Check dataset structure
    with h5py.File(hdf_path, 'r') as hf:
        if prefix in hf:
            prefix_node = hf[prefix]
        elif PREFIX_FALLBACKS[prefix] in hf:
            prefix = PREFIX_FALLBACKS[prefix]
            prefix_node = hf[prefix]
        else:
            raise ValueError(f'Unkown prefix /{prefix}/ in hdf file: {hdf_path}')

        if datasets == 'same':
            if prefix in ('train', 'task'):
                raise ValueError('The flag "same" for dataset specification is only allowed for validation and test.')
            datasets = datasets_cfg.training.dataset
        if datasets == 'all':
            if any(var in prefix_node for var in hdf_vatiables):
                datasets = {'': None}
            else:
                datasets = {d: None for d in prefix_node.keys()}
        elif isinstance(datasets, str):
            if datasets in prefix_node:
                datasets = {datasets: None}
            else:
                datasets = {'': datasets}
        elif isinstance(datasets, (list, tuple)):
            infos = datasets
            datasets = {}
            for info in infos:
                if isinstance(info, dict):
                    datasets.update(info)
                elif isinstance(info, str):
                    datasets[info] = None
        else:
            raise ValueError(f'Invalid dataset value: {datasets}.\n'
                             f'Should be either "all", or a dataset name ')

        missing_datasets = [n for n in datasets if n not in prefix_node]
        if missing_datasets:
            raise ValueError(f'Missing dataset {[prefix_node.name+"/"+n for n in missing_datasets]}'
                             f'in hdf file: {hdf_path}.')
        for dataset_name in datasets:
            check_dataset_exist(prefix_node[dataset_name], hdf_vatiables, hdf_path)

        probed_var = next(iter(hdf_vatiables))
        datasets = [DatasetInfos.from_dataset_infos(prefix_node if path == '' else prefix_node[path],
                                                    probed_var=probed_var, info=infos, rng=seed)
                    for path, infos in datasets.items()]

    generic_hrf_kwargs = dict(path=hdf_path, mapping=fields_mapping, prefix=prefix, cache=cfg['preload-in-RAM'],
                              factor=factor, data_augmentations=data_augmentations, augmented_fields=augmented_fields)
    if prefix in ('testing', 'test'):
        prefix = f'/{prefix}/'
        return {'testing' if d.path == prefix else d.path[len(prefix):]:
                    GenericHRF([d], **generic_hrf_kwargs) for d in datasets}
    return GenericHRF(datasets, **generic_hrf_kwargs)


def check_dataset_exist(hf_node, data_names, hdf_path):
    missing_data_names = {n for n in data_names if n not in hf_node}
    if missing_data_names:
        raise ValueError(f'The following dataset were not found in the hdf archive "{hdf_path}:{hf_node.name}":\n'
                         f' {missing_data_names}')

    invalid_data_names = {n for n in data_names if not isinstance(hf_node[n], h5py.Dataset)}
    if invalid_data_names:
        raise ValueError(f'Thw following node are not dataset in the hdf archive "{hdf_path}:{hf_node.name}":\n'
                         f' {invalid_data_names}')


def extract_variable_from_expr(expr):
    if '{{' not in expr and '}}' not in expr:
        return {expr}
    else:
        vars = {_.split('}}')[0].strip() for _ in expr.split('{{')}
        if '' in vars:
            vars.remove('')
        return vars


class DatasetInfos:
    def __init__(self, path: str, mapping: List[int] = None, slice: slice = None, length: int = None):
        self.path = path
        self.mapping = self.slice = None
        if mapping is not None:
            self.mapping = mapping
            self.length = len(mapping)
        elif slice is not None:
            self.slice = slice
            self.length = (slice.stop-slice.start)//slice.step
        else:
            self.length = length

    def map_index(self, i):
        if self.mapping is not None:
            return self.mapping[i]
        elif self.slice is not None:
            return self.slice.start + i*self.slice.step
        else:
            return i % self.length

    def iter_index(self):
        if self.mapping is not None:
            return self.mapping
        elif self.slice is not None:
            return range(self.slice.start, self.slice.stop, self.slice.step)
        else:
            return range(self.length)

    @property
    def idxs(self):
        if self.mapping is not None:
            return self.mapping
        elif self.slice is not None:
            return self.slice
        else:
            return slice(self.length)

    @staticmethod
    def from_dataset_infos(node: h5py.Group, probed_var: str, info=None, rng: int=1234):
        probed_node = node[probed_var]
        max_length = probed_node.shape[0]
        path = node.name
        if isinstance(info, (list, tuple)):
            idxs = list(int(round(_ % max_length)) for _ in info)
            return DatasetInfos(path, mapping=idxs)
        if info is None:
            return DatasetInfos(path, length=max_length)
        elif isinstance(info, str):
            def to_float(s):
                try:
                    s = s.strip()
                    if s.endswith('%'):
                        s = int(round(float(s[:-1].strip())/100 * max_length))
                    return float(s)
                except ValueError:
                    return None

            proportion = to_float(info)
            if proportion is None:
                info = slice(*(to_float(_) for _ in info.split(':')))
            else:
                info = proportion

        if isinstance(info, (float, int)):
            rng = np.random.default_rng(seed=rng)
            idxs = np.arange(max_length)
            rng.shuffle(idxs)
            idxs = list(int(_) for _ in idxs)

            if info != 0 and -1 < info < 1:
                length = int(round(max_length*info))
            else:
                length = int(round(info) if info>=0 else round(info) % max_length)
            return DatasetInfos(path, mapping=idxs[:length])

        elif isinstance(info, slice):
            start, stop, step = info.start, info.stop, info.step
            if start is None:
                start = 0
            elif start != 0 and -1 < start < 1:
                start = int(round(start * max_length))
            start = int(round(start % max_length))

            if stop is None:
                stop = max_length
            elif stop != 0 and -1 < stop < 1:
                stop = int(round(stop*max_length))
            stop = int(round(stop % max_length))
            if stop == 0:
                stop = max_length

            step = 1 if step is None else int(round(step))

            return DatasetInfos(path, slice=slice(start % max_length, stop % max_length, step))


class GenericHRF(Dataset):
    def __init__(self, datasets: List[DatasetInfos], path: str, mapping: Dict[str, str], prefix: str = None,
                 cache=False,
                 factor=1, data_augmentations: List[DataAugment] = None, augmented_fields: Dict[str, str]=None):
        super(GenericHRF, self).__init__()

        self.datasets = datasets
        self.path = path
        self.prefix = prefix
        self._field2variables = {field: extract_variable_from_expr(expr) for field, expr in mapping.items()}
        self._variables = {_ for v in self._field2variables.values() for _ in v}
        self._variables = {var: f'VAR{i}' for i, var in enumerate(self._variables)}
        self.mapping = deepcopy(mapping)
        for var, name in self._variables.items():
            for field in self.mapping.keys():
                if self.mapping[field].strip() == var:
                    self.mapping[field] = name
                else:
                    self.mapping[field] = self.mapping[field].replace("{{"+var+"}}", name)

        self.cache = cache
        self.factor = factor

        self.data_augmentations = data_augmentations
        if data_augmentations:
            self.augmented_fields = augmented_fields
            augment_mapping = {}
            for field, type in augmented_fields.items():
                augmentable_data_types = data_augmentations[0].augmentable_data_types()
                if type not in augmentable_data_types:
                    if type+'s' in data_augmentations[0].augmentable_data_types():
                        type = type+'s'
                    else:
                        raise TypeError(f'Unkown data-augmentation type {type} for field {field}.')
                if type not in augment_mapping:
                    augment_mapping[type] = set()
                augment_mapping[type].add(field)

            self._augmentations = [da.compile(**augment_mapping,
                                              transpose_input=i == 0,   # Transpose if first
                                              to_torch=i == len(data_augmentations)-1)  # Convert to torch if last
                                   for i, da in enumerate(data_augmentations)]
        else:
            self.augmented_fields = {}
            self._augmentations = []

        self._datasets_length = []
        if cache:
            self._cache = {alias: [] for alias in self._variables.values()}
        else:
            self._hdf_file = None

        with h5py.File(path, 'r') as f:
            for infos in datasets:
                self._datasets_length.append(infos.length)

                if cache:
                    for var, alias in self._variables.items():
                        self._cache[alias] += [np.stack([f[infos.path][var][i] for i in infos.iter_index()])]

    def __len__(self):
        return sum(self._datasets_length) * self.factor

    def open(self):
        if self.cache:
            return False
        if self._hdf_file is None:
            self._hdf_file = h5py.File(self.path, mode='r')
        return self._hdf_file

    def close(self):
        if self.cache or self._hdf_file is None:
            return
        self._hdf_file.close()
        self._hdf_file = None

    def __del__(self):
        self.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def dataset_idx(self, index):
        start_idx = 0
        for dataset_id, length in enumerate(self._datasets_length):
            if index-start_idx < length*self.factor:
                return dataset_id, (index-start_idx)//self.factor

            start_idx += length*self.factor

    def hdf_get_vars(self, index):
        dataset_idx, i = self.dataset_idx(index)
        if self.cache:
            return {k: v[dataset_idx][i] for k, v in self._cache.items()}
        else:
            dataset_infos = self.datasets[dataset_idx]
            f = self.open()
            root_node = f[dataset_infos.path]
            return {alias: root_node[var][dataset_infos.map_index(i)] for var, alias in self._variables.items()}

    def __getitem__(self, i):
        vars = self.hdf_get_vars(i)
        try:
            fields = {f: vars[expr] if expr in vars else
            eval(expr, {'np': np}, vars) for f, expr in self.mapping.items()}
        except NameError as e:
            raise NameError(repr(e) + f'\n VARS: {list(vars.keys())}, map: {self._variables}.')
        for aug in self._augmentations:
            augmented_fields = {f: fields[f] for f in self.augmented_fields.keys()}
            fields.update(aug(**augmented_fields))

        return fields
