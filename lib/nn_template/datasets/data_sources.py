import os

import numpy as np
import pandas as pd

from ..config import Cfg
from ..misc.clip_pad import clip_pad_center


class DataCollectionsAttr(Cfg.multi_type_collection):
    def __init__(self):
        super(DataCollectionsAttr, self).__init__(obj_types={
            'Image': ColorImage,
            'Label2D': Label2D,
            'Mask2D': Mask2D,
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
            raise Cfg.InvalidAttr(f'The source directory for "{self.fullname}" is invalid',
                                  f'Path "{dir}" does not exist.', mark=self.source.mark)
        return dir

    def fetch_indexes(self):
        from .path_utils import PathTemplate

        return pd.DataFrame(PathTemplate(self.path, format_output='dict')
                            .parse_dir(self.dir, recursive=self.search_recursive))


class ImageLoader(FilesPathLoader):
    clip_pad = Cfg.shape(dim=2, default=None)
    clip_pad_center = Cfg.oneOf('center', Cfg.shape(dim=2, default=None), default='center')
    resize = Cfg.shape(dim=2, default=None)
    interpolation = Cfg.oneOf('auto', 'nearest', 'linear', 'area', 'cubic', 'lanczos4', default='auto')

    def fetch_data(self, path):
        import cv2
        img = cv2.imread(path)
        if self.clip_pad:
            img = img.transpose((2, 0, 1))
            center = (0.5, 0.5) if self.clip_pad_center == 'center' else self.clip_pad_center
            img = clip_pad_center(img, center=center, shape=self.clip_pad)
            img = img.transpose((1, 2, 0))
        if self.resize:
            img = cv2.resize(img, self.resize, self.interp_resize)
        return img

    @property
    def interp_resize(self):
        import cv2
        if self.interpolation == 'auto':
            if isinstance(self, (Label2D, Mask2D)):
                return cv2.INTER_AREA
            else:
                return cv2.INTER_CUBIC
        return {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'area': cv2.INTER_AREA,
            'cubic': cv2.INTER_CUBIC,
            'lanczos4': cv2.INTER_LANCZOS4,
        }[self.interpolation]


class ColorImage(ImageLoader):
    def fetch_data(self, path):
        import numpy as np
        img = super(ColorImage, self).fetch_data(path)
        return img.astype(np.float32)/255


class Label2D(ImageLoader):
    labelize = Cfg.collection(str, default={})

    def fetch_data(self, path):
        import cv2
        import numpy as np
        img = super(Label2D, self).fetch_data(path)

        label = np.zeros(img.shape[:2], np.uint8)
        channels = {'img': img, 'b': img[..., 0], 'g': img[..., 1], 'r': img[..., 2]}
        libs = {'np': np, 'cv2': cv2}
        for v, expr in self.labelize.items():
            from ..config.cfg_object import IntAttr
            v = IntAttr.interpret(v)
            mask = eval(expr, libs, channels) & (label == 0)
            label[mask] = v

        return label


class Mask2D(ImageLoader):
    mask: str = 'mean'
    threshold = Cfg.float(min=0, max=1, default=0.5)

    def fetch_data(self, path):
        img = super(Mask2D, self).fetch_data(path)

        if self.mask == 'mean':
            return img.mean(axis=2) > 128*self.threshold
        else:
            import cv2
            import numpy as np
            channels = {'img': img, 'b': img[..., 0], 'g': img[..., 1], 'r': img[..., 2]}
            libs = {'np': np, 'cv2': cv2}
            mask = eval(self.mask, channels, libs)
            if mask.dtype == np.uint8:
                mask = mask > 128*self.threshold
            elif mask.dtype == np.float:
                mask = mask > self.threshold
            return mask
