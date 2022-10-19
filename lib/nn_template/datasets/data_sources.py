import os
from ..config import Cfg


class DataSource(Cfg.Obj):
    def __init__(self):
        pass


class DataAttr(Cfg.multi_type_collection):
    def __init__(self):
        super(DataAttr, self).__init__(obj_types={
            'ColorMap': ColorImage,
            'LabelMap': LabelImage,
            'MaskMap': MaskImage,
        })


class DataSourcesAttr(Cfg.multi_type_collection):
    pass


class DataLoader(Cfg.Obj):

    @property
    def indexes(self):
        if not hasattr(self, '_indexes') or self._indexes is None:
            self.update_indexes()
        return self._indexes

    def update_indexes(self):
        self._indexes = self.fetch_indexes()

    def fetch_indexes(self):
        pass

    def fetch_data(self, index):
        pass

    def __getitem__(self, item):
        return self.fetch_data(self.indexes.iloc[item])

# =======================================================================================


class ImageLoader(DataLoader):
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
        return PathTemplate(self.path, format_output='pandas').parse_dir(self.working_dir)

    def fetch_data(self, path):
        import cv2
        img = cv2.imread(path)
        if self.resize:
            img = cv2.resize(img, self.resize)
        return img


class ColorImage(ImageLoader):
    def fetch_data(self, path):
        import numpy as np
        img = super(ColorImage, self).fetch_data(path)
        return img.astype(np.float32)/255


class LabelImage(ImageLoader):
    def fetch_data(self, path):
        img = super(LabelImage, self).fetch_data(path)
        return img


class MaskImage(ImageLoader):
    def fetch_data(self, path):
        img = super(MaskImage, self).fetch_data(path)
        return img > 1
