import os.path as P

import math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as TorchDataset

from ..config import Cfg
from .data_sources import DataCollectionsAttr
from ..data_augmentation import DataAugmentationCfg


DEFAULT_DATA_PATH = P.join(P.abspath(P.dirname(__file__)), '../../DATA')


class AugmentCfg(Cfg.Obj):
    augmentation: DataAugmentationCfg = Cfg.ref('data-augmentation')
    factor: int = 1


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
            idx = pd.DataFrame()
            for name, src_idx in indexes.items():
                idx = pd.merge(idx, pd.DataFrame({name:src_idx}))
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
    factor = Cfg.int(1)
    range: range = Cfg.range(None)


class DatasetCfg(Cfg.Obj):
    source = Cfg.obj_list(main_key='source', obj_types=DatasetSourceRef)
    augment: AugmentCfg = Cfg.obj(default=None, shortcut='augmentation', nullable=True)
    shuffle = Cfg.oneOf(True, False, 'auto', default='auto')

    def get_indexes(self):
        src_indexes = []
        for sourceRef in self.source:
            source: DataSource = sourceRef.source
            src_len = len(source.indexes)
            start = sourceRef.range.start
            idx = np.arrange(src_len)
            if
        return np.concat([np.stack(np.ones(np.uint), np.arrange for ])
        
    def dataset(self):
        return Dataset(self, fields=self.root()['datasets.fields'])


@Cfg.register_obj('datasets')
class DatasetsCfg(Cfg.Obj):
    fields = Cfg.collection(str)
    sources = Cfg.collection(DataSource)

    train: DatasetCfg = Cfg.obj(shortcut='source')
    validate: DatasetCfg = Cfg.obj(shortcut='source')
    test = Cfg.collection(obj_types=DatasetCfg)


class Dataset(TorchDataset):
    def __init__(self, dataset_cfg: DatasetCfg, fields: dict):
        super(Dataset, self).__init__()
        self.dataset_cfg = dataset_cfg
        self.field_cfg = fields

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

def interpret_range(range, n):
    if range.start is None:
        start = 0
    elif -1 < range.start < 0:
        start = math.floor((1+range.start) * n)
    elif 0 <= range.start <= 1:
        start = math.floor(range.start * n)
    else:
        start = range.start % n

    if range.stop is None:
        stop = n
    elif -1 < range.stop < 0:
        stop = math.floor((1 + range.stop) * n)
    elif 0 <= range.start <= 1:
        stop = math.floor(range.stop * n)
    else:
        stop = range.stop % n

    if range.step is None:
        step = 1
    elif -1 < range.step < 1:
        step = range.step * n
    else:
        step = math.floor(range.step)

    return range(start, stop, step)