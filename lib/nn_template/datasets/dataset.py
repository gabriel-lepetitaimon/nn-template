import os.path as P

import math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as TorchDataset

from ..config import Cfg
from .data_sources import DataCollectionsAttr
from ..data_augmentation import DataAugmentationCfg, DataAugment


DEFAULT_DATA_PATH = P.join(P.abspath(P.dirname(__file__)), '../../DATA')


class AugmentCfg(Cfg.Obj):
    augmentation: DataAugmentationCfg = Cfg.ref('data-augmentation')
    factor: int = 1
    images = Cfg.str(None, nullable=True)
    labels = Cfg.str(None, nullable=True)
    angles = Cfg.str(None, nullable=True)
    vectors = Cfg.str(None, nullable=True)


class DataSource(Cfg.Obj):
    dir_prefix = Cfg.str('./')
    data = DataCollectionsAttr()

    @property
    def indexes(self):
        idx = getattr(self, '_indexes', None)
        if idx is None:
            idx = self.fetch_indexes()
            self._indexes = idx
        return idx

    def fetch_indexes(self):
        indexes = {name: src.fetch_indexes() for name, src in self.data.items()}
        if len(indexes) == 1:
            return pd.DataFrame(indexes)
        else:
            idx = None
            for name, src_idx in indexes.items():
                src_idx = pd.DataFrame({name: src_idx})
                if idx is None:
                    idx = src_idx
                else:
                    idx = pd.merge(idx, src_idx, left_index=True, right_index=True)
            return idx

    def update_indexes(self):
        idx = self.fetch_indexes()
        self._indexes = idx
        return idx

    def get_sample(self, i):
        idx = self.indexes.iloc[i]
        return {name: src.fetch_data(idx[name]) for name, src in self.data.items()}


class DatasetSourceRef(Cfg.Obj):
    source: DataSource = Cfg.ref('datasets.sources')
    factor = Cfg.int(1, min=1)
    range = Cfg.range(None)

    def interpret_range(self, n):
        if self.range is None:
            return range(0, n, 1)
        if self.range.start is None:
            start = 0
        elif -1 < self.range.start < 0:
            start = math.floor((1+self.range.start) * n)
        elif 0 <= self.range.start <= 1:
            start = math.floor(self.range.start * n)
        else:
            start = self.range.start % n

        if self.range.stop is None:
            stop = n
        elif -1 < self.range.stop < 0:
            stop = math.floor((1 + self.range.stop) * n)
        elif 0 <= self.range.start <= 1:
            stop = math.floor(self.range.stop * n)
        else:
            stop = self.range.stop % n

        if self.range.step is None:
            step = 1
        elif -1 < self.range.step < 1:
            step = self.range.step * n
        else:
            step = math.floor(self.range.step)

        return self.range(start, stop, step)


class DatasetCfg(Cfg.Obj):
    source = Cfg.obj_list(main_key='source', obj_types=DatasetSourceRef)
    augment: AugmentCfg = Cfg.obj(default=None, shortcut='augmentation', nullable=True)

    def get_indexes(self):
        src_indexes = []
        for sourceRef in self.source:
            source: DataSource = sourceRef.source
            src_len = len(source.indexes)
            interval = sourceRef.interpret_range(src_len)
            idx = np.arange(interval.start, interval.stop, interval.step, dtype=np.uint32)
            if sourceRef.factor > 1:
                idx = np.repeat(idx, sourceRef.factor)
            src_indexes.append(idx)

        return np.concatenate([np.stack([np.ones((len(idx),), np.uint32)*i, idx], 1)
                               for i, idx in enumerate(src_indexes)], axis=0)
        
    def dataset(self):
        return Dataset(self, fields=self.root()['datasets.fields'])


class DatasetFields(Cfg.Obj):
    images = Cfg.collection(str, default={})
    labels = Cfg.collection(str, default={})
    vectors = Cfg.collection(str, default={})
    angles = Cfg.collection(str, default={})

    def all_fields(self):
        fields = {}
        for f in (self.images, self.labels, self.vectors, self.angles):
            fields.update(f)
        return fields

    @property
    def images_keys(self):
        return list(self.images.keys())

    @property
    def labels_keys(self):
        return list(self.labels.keys())

    @property
    def vectors_keys(self):
        return list(self.vectors.keys())

    @property
    def angles_keys(self):
        return list(self.angles.keys())

@Cfg.register_obj('datasets')
class DatasetsCfg(Cfg.Obj):
    fields: DatasetFields = Cfg.obj()
    sources = Cfg.collection(DataSource)

    train: DatasetCfg = Cfg.obj(shortcut='source')
    validate: DatasetCfg = Cfg.obj(shortcut='source')
    test = Cfg.collection(obj_types=DatasetCfg)


class Dataset(TorchDataset):
    def __init__(self, dataset_cfg: DatasetCfg, fields: DatasetFields):
        super(Dataset, self).__init__()
        self.dataset_cfg = dataset_cfg
        self.fields = fields

        self._idxs = self.dataset_cfg.get_indexes()
        self._srcs = [_.source for _ in self.dataset_cfg.source.list()]

        kwargs = dict(images=fields.images_keys,
                      labels=fields.labels_keys,
                      angles=fields.angles_keys,
                      vectors=fields.vectors_keys,
                      to_torch=True, transpose_input=False)
        if self.dataset_cfg.augment is not None:
            self._augment = self.dataset_cfg.augment.augmentation.data_augment.compile(**kwargs)
        else:
            self._augment = DataAugment().compile(**kwargs)

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, item):
        id_src, id_sample = self._idxs[item]
        sample = self._srcs[id_src].get_sample(id_sample)
        fields = {}
        for field, expr in self.fields.all_fields().items():
            if expr in sample:
                fields[field] = sample[expr]
            else:
                import torch
                fields[field] = eval(expr, {'np': np, 'torch': torch}, sample)
        fields = self._augment(**fields)
        return fields
