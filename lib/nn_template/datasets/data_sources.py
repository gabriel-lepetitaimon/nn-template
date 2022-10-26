import os
from ..config import Cfg


class DataCollectionsAttr(Cfg.multi_type_collection):
    def __init__(self):
        super(DataCollectionsAttr, self).__init__(obj_types={
            'ColorMap': ColorImage,
            'LabelMap': LabelImage,
            'MaskMap': MaskImage,
        })


class DataLoader(Cfg.Obj):

    @property
    def source(self):
        source = self.root(2)
        from .dataset import DataSource
        if isinstance(source, DataSource):
            return source
        return None

    def source_name(self):
        source = self.source
        return source.name if source is not None else ""

    def fetch_indexes(self):
        pass

    def fetch_data(self, index):
        pass

# =======================================================================================


class FilesPathLoader(DataLoader):
    path = Cfg.str('{ID}')
    directory = Cfg.str('')
    search_recursive = Cfg.bool(True)

    @property
    def dir(self):
        dir = './'
        if self.source:
            dir = self.source.get('dir-prefix', dir)
        dir += self.directory
        if not os.path.exists(dir):
            raise Cfg.InvalidAttr(f"The source directory for {self.name} is invalid",
                                  f'Path "{dir}" does not exist.')
        return dir

    def fetch_indexes(self):
        from .path_utils import PathTemplate
        return PathTemplate(self.path, format_output='pandas') \
            .parse_dir(self.dir, recursive=self.search_recursive)


class ImageLoader(FilesPathLoader):
    resize = Cfg.shape(dim=2, default=None, nullable=True)

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
