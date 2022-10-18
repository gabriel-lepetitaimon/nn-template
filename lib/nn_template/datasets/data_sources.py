import os
from ..config import Cfg


class DataSource(Cfg.Obj):
    def __init__(self):
        pass


class DataAttr(Cfg.multi_type_collection):
    def __init__(self):
        super(DataAttr, self).__init__(obj_types={
            'ColorMap': ColorMap,
            'LabelMap': LabelMap,
            'MaskMap': MaskMap,
        })


class DataSourcesAttr(Cfg.multi_type_collection):
    pass


class DataLoader(Cfg.Obj):
    def fetch_indexes(self):
        pass

    def fetch_data(self, index):
        pass

# =======================================================================================


class MapLoader(DataLoader):
    path = Cfg.str()
    resize = Cfg.shape(dim=2, default=None, nullable=True)

    @property
    def working_dir(self):
        cwd = './'
        if self.parent:
            return self.parent.get('path-prefix', cwd)
        return cwd

    def fetch_indexes(self):
        from .path_utils import PathTemplate
        files = PathTemplate(self.path).parse_dir(self.working_dir)

    def fetch_data(self, path):
        pass


class ColorMap(MapLoader):
    def fetch_data(self, path):
        import numpy as np
        img = super(ColorMap, self).fetch_data(path)
        return img.astype(np.float32)/255


class LabelMap(MapLoader):
    def fetch_data(self, path):
        img = super(ColorMap, self).fetch_data(path)
        return img


class MaskMap(MapLoader):
    def fetch_data(self, path):
        img = super(ColorMap, self).fetch_data(path)
        return img > 1
