import os.path as P
from torch.utils.data import Dataset as TorchDataset

from ..config import Cfg
from .data_sources import DataSource, DataSourcesAttr
from ..data_augmentation import DataAugmentationCfg


DEFAULT_DATA_PATH = P.join(P.abspath(P.dirname(__file__)), '../../DATA')


class AugmentCfg(Cfg.Obj):
    augmentation: DataAugmentationCfg = Cfg.ref('data-augmentation')
    factor: int = 1


class DatasetCfg(Cfg.Obj):
    source: DataSource = Cfg.ref('datasets.sources')
    augment: AugmentCfg = Cfg.obj()


class DatasetsCfg(Cfg.Obj):
    fields = Cfg.collection(str)
    sources = DataSourcesAttr()

    train: DatasetCfg = Cfg.obj(shortcut='source')
    validate: DatasetCfg = Cfg.obj(shortcut='source')
    test = Cfg.collection(obj_types=DatasetCfg)

    def datasets(self):
        pass


class Dataset(TorchDataset):
    def __init__(self):
        super(Dataset, self).__init__()

