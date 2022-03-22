import os.path as P
from torch.utils.data import Dataset as TorchDataset

from ..config import Cfg
from .data_sources import DataSources
from ..data_augmentation import CfgDataAugmentation


DEFAULT_DATA_PATH = P.join(P.abspath(P.dirname(__file__)), '../../DATA')


class DataSourcesAttr(Cfg.multi_type_collection):
    def __init__(self):
        super(DataSourcesAttr, self).__init__(
                type_key='type',
                obj_types={'ImageFolders': ...,})


class CfgDataset(Cfg.Obj):
    fields: Cfg.collection(str)
    train: DataSourcesAttr()
    validate: list
    test: list
    sources: DataSourcesAttr()

    def datasets(self):
        pass


class Dataset(TorchDataset):
    def __init__(self):
        super(Dataset, self).__init__()