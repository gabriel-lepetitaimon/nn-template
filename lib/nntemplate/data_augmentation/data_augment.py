import cv2
import functools
import inspect
import numpy as np
import math
from collections import OrderedDict
from copy import copy

import torch

from ..config import Cfg
from .random_dist import RandomDistribution as RD
from .random_dist import RandDistAttr

INTERPOLATIONS = {
    'linear': cv2.INTER_LINEAR,
    'nearest': cv2.INTER_NEAREST,
    'area': cv2.INTER_AREA,
    'cubic': cv2.INTER_CUBIC,
    'max': cv2.INTER_MAX,
    'bits': cv2.INTER_BITS,
}

BORDER_MODES = {
    'constant': cv2.BORDER_CONSTANT,
    'replicate': cv2.BORDER_REPLICATE,
    'wrap': cv2.BORDER_WRAP,
    'reflect': cv2.BORDER_REFLECT,
    'reflect101': cv2.BORDER_REFLECT101,
}


class BorderModeCfg(Cfg.Obj):
    mode = Cfg.strMap(BORDER_MODES, 'constant')
    value = Cfg.float(0)


class RotationCfg(Cfg.Obj):
    enabled: bool = True
    angle = RandDistAttr([-180, 180], symetric=True)
    interpolation = Cfg.strMap(INTERPOLATIONS, 'linear')
    border_mode: BorderModeCfg = Cfg.obj(shortcut='mode', default='constant')


class ElasticCfg(Cfg.Obj):
    enabled: bool = True
    alpha: float = 10
    sigma: float = 20
    alpha_affine: float = 50
    approximate: bool = False
    interpolation = Cfg.strMap(INTERPOLATIONS, 'linear')
    border_mode: BorderModeCfg = Cfg.obj(shortcut='mode', default='constant')


class CropCfg(Cfg.Obj):
    shape = Cfg.shape(None, dim=2)
    padding = Cfg.shape(0, dim=2)


@Cfg.register_obj('data-augmentation', collection='default')
class DataAugmentationCfg(Cfg.Obj):
    flip = Cfg.oneOf(False, True, 'horizontal', 'vertical', default=False)
    rotation: RotationCfg = Cfg.obj(default=False, shortcut='enabled')
    rot90 = False
    elastic: ElasticCfg = Cfg.obj(default=False, shortcut='enabled')
    random_crop: CropCfg = Cfg.obj(default=None, shortcut='shape')

    gamma = RandDistAttr(default=None, symetric=True)
    brightness = RandDistAttr(default=None, symetric=True)
    contrast = RandDistAttr(default=None, symetric=True)

    hue = RandDistAttr(default=None, symetric=True)
    saturation = RandDistAttr(default=None, symetric=True)
    value = RandDistAttr(default=None, symetric=True)

    seed: int = 1234

    @functools.cached_property
    def data_augment(self):
        return self.create_data_augment()

    def create_data_augment(self):
        da = DataAugment()

        match self.flip:
            case True: da.flip()
            case 'horizontal': da.flip_horizontal()
            case 'vertical': da.flip_vertical()

        patch_shape = None
        if self.random_crop and self.random_crop.shape:
            patch_shape = np.array(self.random_crop.shape)
            if self.rotation.enabled:
                patch_shape = patch_shape*np.sqrt(2)
            if self.elastic.enabled:
                patch_shape = patch_shape + max(3*self.elastic.alpha, 3*self.elastic.sigma)
            patch_shape = tuple(int(math.ceil(_)) for _ in patch_shape)
            da.crop(shape=patch_shape, padding=self.random_crop.padding)
            if patch_shape != self.random_crop.shape:
                patch_shape = self.random_crop.shape

        if self.rotation.enabled:
            rot = self.rotation
            da.rotate(angle=rot.angle, interpolation=rot.interpolation,
                      border_mode=rot.border_mode.mode, border_value=rot.border_mode.value)

        if self.elastic.enabled:
            ela = self.elastic
            da.elastic_distortion(alpha=ela.alpha, alpha_affine=ela.alpha_affine, sigma=ela.sigma,
                                  approximate=ela.approximate, interpolation=ela.interpolation,
                                  border_mode=ela.border_mode.mode, border_value=ela.border_mode.value)

        if patch_shape:
            da.crop_center(shape=patch_shape)

        if self.rot90:
            da.rot90()

        if self.gamma or self.brightness or self.contrast:
            da.color(brightness=self.brightness, gamma=self.gamma, contrast=self.contrast)

        if self.hue or self.saturation or self.value:
            da.hsv(hue=self.hue, saturation=self.saturation, value=self.value)

        return da


########################################################################################################################
#                   ---  DATA AUGMENTATION  ---
########################################################################################################################


_augment_methods = {}
_augment_by_type = {}


def augment_method(augment_type=None):
    def decorator(func):
        @functools.wraps(func)
        def register_augment(self, *params, **kwargs):
            params = bind_args(func, params, kwargs)
            self._augment_stack.append((func.__name__, params))
            return self

        _augment_methods[func.__name__] = func, augment_type
        _augment_by_type[augment_type] = func.__name__
        return register_augment

    return decorator


class DataAugment:
    def __init__(self, seed=1234):
        self._augment_stack = []
        self._rng = np.random.default_rng(seed)

    class Augmenter:
        def __init__(self, da, images='', labels='', angles='', vectors='', deaugment=False,
                     to_torch=False, transpose_input=False, rng=None):
            self.da = da
            if rng is None:
                self.rng = self._rng
            elif isinstance(rng, int):
                self.rng = np.random.default_rng(rng)

            self.to_torch = to_torch
            self.transpose_input = transpose_input

            if isinstance(images, str):
                self.images = [_.strip() for _ in images.split(',') if _.strip()]
            elif not images:
                self.images = ()
            if isinstance(labels, str):
                self.labels = [_.strip() for _ in labels.split(',') if _.strip()]
            elif not labels:
                self.labels = ()
            if isinstance(angles, str):
                self.angles = [_.strip() for _ in angles.split(',') if _.strip()]
            elif not angles:
                self.angles = ()
            if isinstance(vectors, str):
                self.vectors = [_.strip() for _ in vectors.split(',') if _.strip()]
            elif not vectors:
                self.vectors = ()

            self.images_aug = None
            self.labels_aug = None
            self.angles_aug = None
            self.fields_aug = None

            self.rng_states_def = []
            if images:
                self.images_aug = self.da.compile_stack(rng_states=self.rng_states_def)
            if labels:
                self.labels_aug = self.da.compile_stack(rng_states=self.rng_states_def,
                                                        interpolation=cv2.INTER_NEAREST, except_type={'color'})
            if angles:
                self.angles_aug = self.da.compile_stack(rng_states=self.rng_states_def, border_mode=cv2.BORDER_REPLICATE,
                                                        except_type={'color'}, value_type='angle')
            if vectors:
                self.fields_aug = self.da.compile_stack(rng_states=self.rng_states_def, border_mode=cv2.BORDER_REPLICATE,
                                                        except_type={'color'}, value_type='vec')

        def augment(self, rng_states=None, **kwargs):
            if not isinstance(rng_states, list):
                rng = self.rng if rng_states is None else rng_states
                rng_states = [[s(rng) for s in states.values()] for states in self.rng_states_def]

            data = copy(kwargs)
            for k, v in data.items():
                if self.transpose_input and v.ndim == 3:
                    data[k] = v.transpose(1, 2, 0)

            for image in self.images:
                data[image] = self.images_aug(data[image], rng_states)

            if self.labels:
                mixed_label = sum(data[label]*(4**i) for i, label in enumerate(self.labels))
                mixed_label = self.labels_aug(mixed_label, rng_states)
                for i, label in enumerate(self.labels):
                    data[label] = (mixed_label//(4**i)) % 4

            for angle in self.angles:
                data[angle] = self.angles_aug(data[angle], rng_states)

            for field in self.vectors:
                data[field] = self.fields_aug(data[field], rng_states)

            if self.to_torch:
                data = {k: to_tensor(v) for k, v in data.items()}
                for label in self.labels:
                    data[label] = data[label].long()

            return data

    def compile(self, images='', labels='', angles='', vectors='', to_torch=False, transpose_input=False, rng=None):
        if rng is None:
            rng = self._rng
        elif isinstance(rng, int):
            rng = np.random.default_rng(rng)
        if isinstance(images, str):
            images = [_.strip() for _ in images.split(',') if _.strip()]
        elif not images:
            images = ()
        if isinstance(labels, str):
            labels = [_.strip() for _ in labels.split(',') if _.strip()]
        elif not labels:
            labels = ()
        if isinstance(angles, str):
            angles = [_.strip() for _ in angles.split(',') if _.strip()]
        elif not angles:
            angles = ()
        if isinstance(vectors, str):
            vectors = [_.strip() for _ in vectors.split(',') if _.strip()]
        elif not vectors:
            vectors = ()

        images_f = None
        labels_f = None
        angles_f = None
        fields_f = None

        rng_states_def = []
        if images:
            images_f = self.compile_stack(rng_states=rng_states_def)
        if labels:
            labels_f = self.compile_stack(rng_states=rng_states_def,
                                          interpolation=cv2.INTER_NEAREST, except_type={'color'})
        if angles:
            angles_f = self.compile_stack(rng_states=rng_states_def, border_mode=cv2.BORDER_REPLICATE,
                                          except_type={'color'}, value_type='angle')
        if vectors:
            fields_f = self.compile_stack(rng_states=rng_states_def, border_mode=cv2.BORDER_REPLICATE,
                                          except_type={'color'}, value_type='vec')

        def augment(rng=rng, **kwargs):
            rng_states = [[s(rng) for s in states.values()] for states in rng_states_def]

            data = copy(kwargs)
            for k, v in data.items():
                if transpose_input and v.ndim == 3:
                    data[k] = v.transpose(1, 2, 0)

            for image in images:
                data[image] = images_f(data[image], rng_states)

            if labels:
                mixed_label = sum(data[label]*(4**i) for i, label in enumerate(labels))
                mixed_label = labels_f(mixed_label, rng_states)
                for i, label in enumerate(labels):
                    data[label] = (mixed_label//(4**i)) % 4

            for angle in angles:
                data[angle] = angles_f(data[angle], rng_states)

            for field in vectors:
                data[field] = fields_f(data[field], rng_states)

            if to_torch:
                return {k: to_tensor(v) for k, v in data.items()}

            return data

        return augment

    @staticmethod
    def augmentable_data_types():
        from inspect import signature
        return {arg for arg in signature(DataAugment.compile).parameters if arg not in ('to_str', 'rng')}

    def compile_stack(self, rng_states, only_type='', except_type='', **kwargs):
        """

        Args:
            rng_states: reference to the global rng_states.
            only_type:
            except_type:
            **kwargs:

        Returns:

        """

        only_type = str2tuple(only_type)
        except_type = str2tuple(except_type)
        
        # --- Prepare functions stack ---
        f_stack = []
        if rng_states is None:
            rng_states = []
        if len(rng_states) == 0:
            rng_states += [OrderedDict() for _ in range(len(self._augment_stack))]

        for f_i, (f_name, f_params) in enumerate(self._augment_stack):
            f, a_type = _augment_methods[f_name]

            if a_type is not None and (a_type in except_type or (only_type and a_type not in only_type)):
                continue

            # Rerun the function generating augment() with the custom provided parameters
            params = bind_args_partial(f, kwargs=kwargs)
            f_params.update(params)
            f_augment, *rng_params = match_params(f, self=self, **f_params)

            if isinstance(f_augment, dict):
                f_pre = f_augment.get('pre', None)
                f_post = f_augment.get('post', None)
                f_augment = f_augment['augment']
            else:
                f_pre = None
                f_post = None

            # Read the name of the random parameters needed by augment()
            rng_params_names = list(inspect.signature(f_augment).parameters.keys())[1:]
            assert len(rng_params_names) == len(rng_params), f'Invalid random parameters count for ' \
                                                              f'augmentation function: {f_name}(**{params}).'
            for p_name, p_dist in zip(rng_params_names, rng_params):
                if p_name in rng_states[f_i]:
                    if rng_states[f_i][p_name] != p_dist:
                        raise ValueError(f'Inconsistent distribution for the random parameter {p_name} of the '
                                         f'augment function {f_name}: \n'
                                         f'{rng_states[f_i][p_name]} != {p_dist}')
                    else:
                        continue
                rng_states[f_i][p_name] = p_dist
            f_stack.append((f_i, f_pre, f_augment, f_post))

        def augment(x, rng_state):
            reduce = False
            if x.ndim == 2:
                reduce = True
                x = x[:, :, np.newaxis]
            elif x.ndim != 3:
                raise ValueError('Invalid cv image format, shape is: %s' % repr(x.shape))
            h, w, c = x.shape
            x_cv = []
            for i in range(c//3):
                x_cv += [x[..., i*3:(i+1)*3]]
            for i in range(c-(c % 3), c):
                x_cv += [x[..., i:i+1]]

            for f_i, f_pre, f_augment, f_post in f_stack:
                f_params = list(rng_state[f_i])
                if f_pre is not None:
                    x_cv = f_pre(x_cv, *f_params)
                x_cv = [f_augment(x, *f_params) for x in x_cv]
                if f_post is not None:
                    x_cv = f_post(x_cv, *f_params)

            x_cv = np.concatenate(x_cv, axis=2)
            if reduce and x_cv.shape[2]==1:
                x_cv = x_cv[:,:,0]
            return x_cv
        return augment

    @augment_method('geometric')
    def shear(self, x=(-5,5), y=(-5,5), value_type=None):
        """
        Perform data augmentation by shearing the image.
        :param x: Angle of horizontal shear in degrees in range [-90, 90].
        :param y: Angle of vertical shear in degrees.
        :param value_type:
        :return:
        """
        x = RD.auto(x, symetric=True)
        y = RD.auto(y, symetric=True)

        def augment(x, x_shear, y_shear):
            h, w, _ = x.shape
            x_shear = np.tan(x_shear*np.pi/180)
            y_shear = np.tan(y_shear*np.pi/180)
            M = np.array([[1, x_shear, 0],
                          [y_shear, 1, 0]])
            x = cv2.warpAffine(x, M, (w, h))
            return x

        return augment, x, y

    @augment_method('geometric')
    def flip(self, p_horizontal=0.5, p_vertical=0.5, value_type=None):
        h_flip = RD.binary(p_horizontal)
        v_flip = RD.binary(p_vertical)

        post_flip = None
        if value_type == 'angle':
            def post_flip(X, h, v):
                if h:
                    X = [-x for x in X]
                if v:
                    X = [np.pi-x for x in X]
                if v or h:
                    X = [x%(2*np.pi) for x in X]
                return X
        elif value_type == 'vec':
            def post_flip(X, h, v):
                x, y = X
                return -x if v else x, -y if h else y

        def augment(x, h, v):
            if h:
                x = np.flip(x, axis=1)
            if v:
                x = np.flip(x, axis=0)
            return x
        return {'augment': augment, 'post': post_flip,
                'deaugment': augment, 'post_deaugment': post_flip}, h_flip, v_flip

    def flip_horizontal(self, p=0.5):
        return self.flip(p_horizontal=p, p_vertical=0)

    def flip_vertical(self, p=0.5):
        return self.flip(p_horizontal=0, p_vertical=p)

    @augment_method('geometric')
    def rot90(self, value_type=None):
        rot90 = RD.discrete_uniform(4)

        post_rot90 = None
        if value_type == 'angle':
            def post_rot90(X, k):
                return [x+k*np.pi/2 for x in X]
        elif value_type == 'vec':
            def post_rot90(X, k):
                k %= 4
                u, v = X
                if k == 0:
                    return X
                elif k == 1:
                    return -v, u
                elif k == 2:
                    return -u, -v
                else:
                    return v, -u

        def augment(x, k):
            return np.rot90(x, k=k, axes=(0, 1))

        def deaugment(x, k):
            return np.rot90(x, k=-k, axes=(0, 1))

        post_deaugment = None if post_rot90 is None else lambda x, k: post_rot90(x, -k)

        return {'augment': augment, 'post': post_rot90,
                'deaugment': deaugment, 'post_deaugment': post_deaugment}, rot90

    @augment_method('geometric')
    def rotate(self, angle=(-180, +180), value_type=None,
               interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0):
        import albumentations.augmentations.geometric.functional as AF
        angle = RD.auto(angle, symetric=True)

        pre_rot = None
        if value_type == 'angle':
            def pre_rot(X, phi):
                return [np.cos(x+(phi*np.pi/180)) for x in X]+[np.sin(x+(phi*np.pi/180)) for x in X]
        elif value_type == 'vec':
            def pre_rot(X, phi):
                u, v = X
                phi = phi*np.pi/180
                return (u*np.cos(phi) - v*np.sin(phi),
                        v*np.cos(phi) + u*np.sin(phi))

        def augment(x, angle):
            return AF.rotate(x, angle, interpolation=interpolation, border_mode=border_mode, value=border_value)

        post_rot = None
        if value_type == 'angle':
            def post_rot(X, phi):
                N = len(X)//2
                return [np.arctan2(cos, sin) for cos, sin in zip(X[:N], X[N:])]
        
        return {'pre': pre_rot, 'augment': augment, 'post': post_rot}, angle

    @augment_method('geometric')
    def elastic_distortion(self, alpha=1, sigma=50, alpha_affine=50,
                           approximate=False, interpolation=cv2.INTER_LINEAR,
                           border_mode=cv2.BORDER_CONSTANT, border_value=0.0):
        import albumentations.augmentations.geometric.functional as AF
        def augment(x, rng_seed):
            random_state = np.random.RandomState(rng_seed)
            return AF.elastic_transform(x, alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, approximate=approximate,
                                        interpolation=interpolation, border_mode=border_mode, value=border_value,
                                        random_state=random_state)
        return augment, RD.integers(1e8)

    @augment_method('geometric')
    def crop(self, shape, padding=(0, 0)):
        padding = [s-p*2 for s, p in zip(shape, padding)]

        def augment(x, centerX, centerY):
            center = [int(c*(s-p) + p//2)
                      for s, p, c in zip(x.shape, padding, (centerY, centerX))]
            return crop_pad(x, center, shape)
        return augment, RD.uniform(1), RD.uniform(1)

    @augment_method('geometric')
    def crop_center(self, shape):
        def augment(x):
            center = [s//2 for s in x.shape[:2]]
            return crop_pad(x, center, shape)

        return augment,

    @augment_method('color')
    def color(self, brightness=None, contrast=None, gamma=None, r=None, g=None, b=None):
        brightness = RD.constant(0) if brightness is None else RD.auto(brightness, symetric=True)
        contrast = RD.constant(0) if contrast is None else RD.auto(contrast, symetric=True)
        gamma = RD.gamma(0) if gamma is None else RD.auto(gamma, symetric=True)

        r = RD.constant(0) if r is None else RD.auto(r, symetric=True)
        g = RD.constant(0) if g is None else RD.auto(g, symetric=True)
        b = RD.constant(0) if b is None else RD.auto(b, symetric=True)

        def augment(x, brightness, contrast, gamma, r, g, b):
            x = ((x+brightness)*(contrast+1.)).clip(0)**(gamma+1.)
            
            if r or b or g:
                n = x.shape[0]//3
                bgr = np.array([b, g, r]*n)
                x[..., :3*n] = x[..., :3*n] + bgr[np.newaxis, np.newaxis, :]
            return np.clip(x, a_min=0, a_max=1)
        return augment, brightness, contrast, gamma, r, g, b

    def brightness(self, brightness=(-0.1, 0.1)):
        return self.color(brightness=brightness)

    def contrast(self, contrast=(-0.1, 0.1)):
        return self.color(contrast=contrast)

    def gamma(self, gamma=(-0.1, 0.1)):
        return self.color(gamma=gamma)

    @augment_method('color')
    def hsv(self, hue=None, saturation=None, value=None):
        hue = RD.constant(0) if hue is None else RD.auto(hue, symetric=True)
        saturation = RD.constant(0) if saturation is None else RD.auto(saturation, symetric=True)
        value = RD.constant(0) if value is None else RD.auto(value, symetric=True)

        a_min = np.array([0, 0, 0], np.float32)
        a_max = np.array([360, 1, 1], np.float32)

        def augment(x, h, s, v):
            hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
            hsv += np.array([h, s, v], dtype=np.float32)
            hsv[:, :, 0] = hsv[:, :, 0] % 360
            hsv = np.clip(hsv, a_min=a_min, a_max=a_max)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return augment, hue, saturation, value

    def hue(self, hue=(-20, 20)):
        return self.hsv(hue=hue)

    def saturation(self, saturation=(-20, 20)):
        return self.hsv(saturation=saturation)

    @augment_method('intensity')
    def noise(self, noise):
        noise = RD.auto(noise, symetric=True)

        def augment(x, n):
            return x + n

        return augment, noise


def crop_pad(img, center, size):
    y, x = center
    h, w = size
    H, W = img.shape[:2]
    half_h, half_w = (_ // 2 for _ in size)

    y0 = int(max(0, half_h - y))
    y1 = int(max(0, y - half_h))
    h = int(min(h-y0, H-y1))

    x0 = int(max(0, half_w - x))
    x1 = int(max(0, x - half_w))
    w = int(min(w-x0, W-x1))

    r = np.zeros_like(img, shape=tuple(size)+img.shape[2:])
    r[y0:y0+h, x0:x0+w] = img[y1:y1+h, x1:x1+w]
    return r


########################################################################################################################
def bind_args(f, args=(), kwargs=None):
    bind = bind_args_partial(f, args, kwargs)
    missing_args = set(not_optional_args(f)).difference(bind.keys())
    missing_args.difference_update({'self'})
    if missing_args:
        raise ValueError("%s() missing %i required arguments: '%s'"
                         % (f.__name__, len(missing_args), "', '".join(missing_args)))
    return bind


def bind_args_partial(f, args=(), kwargs=None):
    from collections import OrderedDict
    if kwargs is None:
        kwargs = {}
    params = list(inspect.signature(f).parameters.keys())
    bind = OrderedDict()
    for i, a in enumerate(args):
        if params[i] in kwargs:
            raise ValueError("%s() got multiple value for argument '%s'" % (f.__name__, params[i]))
        bind[params[i]] = a
    for k, a in kwargs.items():
        bind[k] = a
    return bind


def match_params(method, args=None, **kwargs):
    """
    Call the specified method, matching the arguments it needs with those,
    provided in kwargs. The useless arguments are ignored.
    If some not optional arguments is missing, a ValueError exception is raised.
    :param method: The method to call
    :param kwargs: Parameters provided to method
    :return: Whatever is returned by method (might be None)
    """
    method_params = inspect.signature(method).parameters.keys()
    method_params = {_: kwargs[_] for _ in method_params & kwargs.keys()}

    if args is None:
        args = []
    i_args = 0
    for not_opt in not_optional_args(method):
        if not_opt not in method_params:
            if i_args < len(args):
                method_params[not_opt] = args[i_args]
                i_args += 1
            else:
                raise ValueError('%s is not optional to call method: %s.' % (not_opt, method))

    return method(**method_params)


def not_optional_args(f):
    """
    List all the parameters not optional of a method
    :param f: The method to analise
    :return: The list of parameters
    :rtype: list
    """
    sig = inspect.signature(f)
    return [p_name for p_name, p in sig.parameters.items()
            if isinstance(inspect._empty, type(p.default)) and inspect._empty == p.default]


def str2tuple(v):
    if isinstance(v, str):
        return tuple(_.strip() for _ in v if _.strip())
    return v


def to_tensor(x):
    import torch
    if x.ndim == 3:
        x = x.transpose(2, 0, 1)
    return torch.from_numpy(x)
