import os
import os.path as P

import math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader

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

    def _init_after_populate(self):
        self.fetch_indexes()

    @property
    def indexes(self):
        idx = getattr(self, '_indexes', None)
        if idx is None:
            idx = self.fetch_indexes()
            self._indexes = idx
        return idx

    @property
    def length(self):
        return len(self.indexes)

    def fetch_indexes(self):
        indexes = {name: src.fetch_indexes() for name, src in self.data.items()}
        if len(indexes) == 1:
            name, indexes = next(iter(indexes.items()))
            return pd.DataFrame({name: indexes['fullpath']})
        else:
            idx = None
            for name, src_idx in indexes.items():
                if not len(src_idx):
                    raise Cfg.InvalidAttr(f'Invalid data source specifications for "{self.fullname}.data.{name}"',
                                          f'No data was provided by this source.', mark=self.data.get_mark(name),)
                src_idx = pd.DataFrame({name: src_idx['fullpath']} |
                                       {'_#_'+ID: src_idx[ID] for ID in src_idx.columns if ID != 'fullpath'})
                if idx is None:
                    idx = src_idx
                else:
                    try:
                        idx = pd.merge(idx, src_idx)
                    except pd.errors.MergeError:
                        idx = pd.merge(idx, src_idx, how='cross')
            return idx.drop(columns=[_ for _ in idx.columns if _.startswith('_#_')])

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
            start = math.floor(self.range.start) % n

        if self.range.stop is None:
            stop = n
        elif -1 < self.range.stop < 0:
            stop = math.floor((1 + self.range.stop) * n)
        elif 0 <= self.range.stop <= 1:
            stop = math.floor(self.range.stop * n)
        else:
            stop = math.floor(self.range.stop) % n

        if self.range.step is None:
            step = 1
        elif -1 < self.range.step < 1:
            step = math.floor(self.range.step * n)
        else:
            step = math.floor(self.range.step)

        return slice(start, stop, step)


class DatasetCfg(Cfg.Obj):
    source = Cfg.obj_list(main_key='source', obj_types=DatasetSourceRef)
    ignore_metrics = Cfg.strList(default=[])
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

    def sample_count(self):
        l = len(self.get_indexes())
        return l * self.augment.factor if self.augment else l


class DatasetFields(Cfg.Obj):
    images = Cfg.collection(str, default={})
    labels = Cfg.collection(str, default={})
    masks = Cfg.collection(str, default={})
    vectors = Cfg.collection(str, default={})
    angles = Cfg.collection(str, default={})

    def all_fields(self):
        fields = {}
        for f in (self.images, self.labels, self.masks, self.vectors, self.angles):
            fields.update(f)
        return fields

    @property
    def images_keys(self):
        return list(self.images.keys())

    @property
    def labels_keys(self):
        return list(self.labels.keys())

    @property
    def masks_keys(self):
        return list(self.masks.keys())

    @property
    def vectors_keys(self):
        return list(self.vectors.keys())

    @property
    def angles_keys(self):
        return list(self.angles.keys())


########################################################################################################################
@Cfg.register_obj('datasets')
class DatasetsCfg(Cfg.Obj):
    minibatch = Cfg.int()
    fields: DatasetFields = Cfg.obj()
    sources = Cfg.collection(DataSource)

    train: DatasetCfg = Cfg.obj(shortcut='source')
    validate: DatasetCfg = Cfg.obj(shortcut='source')
    test = Cfg.collection(obj_types=DatasetCfg)

    @property
    def minibatch_size(self):
        return math.ceil(self.minibatch / self.root().get('hardware.minibatch-splits', 1)) if self.minibatch else None

    def create_train_val_dataloaders(self):
        batch_size = self.minibatch_size
        num_workers = self.root().get('hardware.num_workers', os.cpu_count())

        train = DataLoader(self.train.dataset(),
                           pin_memory=True, shuffle=True,
                           batch_size=batch_size, num_workers=num_workers)
        validate = DataLoader(self.validate.dataset(),
                              pin_memory=True,
                              num_workers=6, batch_size=self.minibatch_size)
        return train, validate

    def create_test_dataloaders(self):
        return [DataLoader(d.dataset(), pin_memory=True,
                           num_workers=6, batch_size=self.minibatch_size)
                for d in self.test.values()]

    @property
    def test_dataloaders_names(self) -> tuple[str]:
        return tuple(self.test.keys())

    def create_all_dataloaders(self):
        train, val = self.create_train_val_dataloaders()
        test = self.create_test_dataloaders()
        return train, val, test


class Dataset(TorchDataset):
    def __init__(self, dataset_cfg: DatasetCfg, fields: DatasetFields):
        super(Dataset, self).__init__()
        self.dataset_cfg = dataset_cfg
        self.fields = fields

        self._idxs = self.dataset_cfg.get_indexes()
        self._srcs = [_.source for _ in self.dataset_cfg.source.list()]

        kwargs = dict(images=fields.images_keys,
                      labels=fields.labels_keys+fields.masks_keys,
                      angles=fields.angles_keys,
                      vectors=fields.vectors_keys,
                      to_torch=True, transpose_input=False)
        if self.dataset_cfg.augment is not None:
            self._augment = self.dataset_cfg.augment.augmentation.data_augment.compile(**kwargs)
        else:
            self._augment = DataAugment().compile(**kwargs)

    def __len__(self):
        l = len(self._idxs)
        return l * self.dataset_cfg.augment.factor if self.dataset_cfg.augment else l

    def __getitem__(self, item):
        id_src, id_sample = self._idxs[item % len(self._idxs)]
        sample = self._srcs[id_src].get_sample(id_sample)
        fields = {}
        for field, expr in self.fields.all_fields().items():
            if expr in sample:
                fields[field] = sample[expr]
            else:
                import torch
                fields[field] = eval(expr, {'np': np, 'torch': torch}, sample)
        fields = self._augment(**fields)
        for f in self.fields.masks_keys:
            fields[f] = fields[f] > 0
        for f in self.fields.labels_keys:
                fields[f] = fields[f].long()
        return fields
