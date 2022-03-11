import cv2
import functools
import inspect
import numpy as np
from collections import OrderedDict
from copy import copy

from steered_cnn.utils import AttributeDict
from .cv2_parse import INTERPOLATIONS, BORDER_MODES

_augment_methods = {}
_augment_by_type = {}


def parse_data_augmentations(cfg: AttributeDict, rng=None):
    if not 'data-augmentation' in cfg:
        return {}
    try:
        rng_seed = int(cfg.training.seed)
    except (TypeError, KeyError):
        rng_seed = rng
    cfg = cfg['data-augmentation'].copy()
    cfg.update({'default': cfg.pop({'flip', 'rotation', 'elastic', 'gamma', 'brightness', 'hue', 'saturation'})})
    return {k: parse_data_augmentation_cfg(v, rng=rng_seed) for k, v in cfg.items()}


def parse_data_augmentation_cfg(cfg: AttributeDict, rng=None):
    da = DataAugment(seed=rng)


    flip = cfg.get('flip', False)
    if flip is True:
        da.flip()
    elif flip=='horizontal':
        da.flip_horizontal()
    elif flip=='vertical':
        da.flip_vertical()
    elif isinstance(flip, AttributeDict):
        flip_h, flip_v = flip.get('horizontal', 0), flip.get('vertical', 0)
        if flip_h:
            da.flip_horizontal(flip_h)
        if flip_v:
            da.flip_horizontal(flip_v)

    rotation = cfg.get('rotation', False)
    if rotation is True:
        da.rotate()
    elif isinstance(rotation, (list, tuple)):
        if len(rotation) != 2:
            raise ValueError(f'Expected min and max value for gamma but got: {rotation}.')
        da.rotate(angle=(rotation[0], rotation[1]))
    elif isinstance(rotation, AttributeDict):
        angle = rotation.get('angle', (-180, 180))
        interpolation = INTERPOLATIONS[rotation.get('interpolation', 'linear')]
        border_mode = rotation.get('border-mode', 'constant')
        if isinstance(border_mode, (dict, AttributeDict)):
            border_value = list(border_mode.values())[0]
            border_mode = list(border_mode.keys())[0]
        else:
            border_value = 0
        border_mode = BORDER_MODES[border_mode]
        da.rotate(angle=angle, interpolation=interpolation, border_mode=border_mode, border_value=border_value)

    elastic = cfg.get('elastic', False)
    if elastic is True:
        da.elastic_distortion(alpha=10, sigma=20, alpha_affine=50)
    elif isinstance(elastic, AttributeDict):
        alpha = elastic.get('alpha', 10)
        sigma = elastic.get('sigma', 20)
        alpha_affine = elastic.get('alpha-affine', 20)
        approximate = elastic.get('approximate', False)
        interpolation = INTERPOLATIONS[elastic.get('interpolation', 'linear')]
        border_mode = elastic.get('border-mode', 'constant')
        if isinstance(border_mode, (dict, AttributeDict)):
            border_value = list(border_mode.values())[0]
            border_mode = list(border_mode.keys())[0]
        else:
            border_value = 0
        border_mode = BORDER_MODES[border_mode]
        da.elastic_distortion(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, approximate=approximate,
                              interpolation=interpolation, border_mode=border_mode, border_value=border_value)

    gamma = cfg.get('gamma', None)
    if gamma is True:
        gamma = (-0.1, 0.1)
    brightness = cfg.get('brightness', None)
    if brightness is True:
        brightness = (-0.1, 0.1)
    if brightness or gamma:
        da.color(brightness=brightness, gamma=gamma)

    hue = cfg.get('hue', None)
    if hue is True:
        hue = (-20, 20)
    saturation = cfg.get('saturation', None)
    if saturation is True:
        saturation = (-20, 20)
    if hue or saturation:
        da.hsv(hue=hue, saturation=saturation)

    return da


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

    def compile(self, images='', labels='', angles='', vectors='', to_torch=False, transpose_input=False, rng=None):
        if rng is None:
            rng = self._rng
        elif isinstance(rng, int):
            rng = np.random.default_rng(rng)
        if isinstance(images, str):
            images = [_.strip() for _ in images.split(',') if _.strip()]
        if isinstance(labels, str):
            labels = [_.strip() for _ in labels.split(',') if _.strip()]
        if isinstance(angles, str):
            angles = [_.strip() for _ in angles.split(',') if _.strip()]
        if isinstance(vectors, str):
            vectors = [_.strip() for _ in vectors.split(',') if _.strip()]

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
                def to_tensor(x):
                    import torch
                    if x.ndim == 3:
                        x = x.transpose(2, 0, 1)
                    return torch.from_numpy(x)
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
            f_augment, *rng_params = match_params(f, self=self, **params)

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
    def flip(self, p_horizontal=0.5, p_vertical=0.5, value_type=None):
        h_flip = _RD.binary(p_horizontal)
        v_flip = _RD.binary(p_vertical)

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
        return {'augment': augment, 'post': post_flip}, h_flip, v_flip

    def flip_horizontal(self, p=0.5):
        return self.flip(p_horizontal=p, p_vertical=0)

    def flip_vertical(self, p=0.5):
        return self.flip(p_horizontal=0, p_vertical=p)

    @augment_method('geometric')
    def rot90(self, value_type=None):
        rot90 = _RD.discrete_uniform(4)

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

        return {'augment': augment, 'post': post_rot90}, rot90

    @augment_method('geometric')
    def rotate(self, angle=(-180, +180), value_type=None,
               interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0):
        import albumentations.augmentations.geometric.functional as AF
        angle = _RD.auto(angle, symetric=True)

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
        return augment, _RD.integers(1e8)

    @augment_method('color')
    def color(self, brightness=None, contrast=None, gamma=None, r=None, g=None, b=None):
        brightness = _RD.constant(0) if brightness is None else _RD.auto(brightness, symetric=True)
        contrast = _RD.constant(0) if contrast is None else _RD.auto(contrast, symetric=True)
        gamma = _RD.gamma(0) if gamma is None else _RD.auto(gamma, symetric=True)

        r = _RD.constant(0) if r is None else _RD.auto(r, symetric=True)
        g = _RD.constant(0) if g is None else _RD.auto(g, symetric=True)
        b = _RD.constant(0) if b is None else _RD.auto(b, symetric=True)

        def augment(x, brightness, contrast, gamma, r, g, b):
            x = (x+brightness)*(contrast+1.).clip(0)**(gamma+1.)
            
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
        hue = _RD.constant(0) if hue is not None else _RD.auto(hue, symetric=True)
        saturation = _RD.constant(0) if saturation is not None else _RD.auto(saturation, symetric=True)
        value = _RD.constant(0) if value is not None else _RD.auto(value, symetric=True)

        a_min = np.array([0, 0, 0], np.uint8)
        a_max = np.array([179, 255, 255], np.uint8)

        def augment(x, h, s , v):
            hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
            hsv = hsv + np.array([h, s, v])
            hsv[:, :, 0] = hsv[:, :, 0] % 179
            hsv = np.clip(hsv, a_min=a_min, a_max=a_max).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return augment, hue, saturation, value

    def hue(self, hue=(-20, 20)):
        return self.hsv(hue=hue)

    def saturation(self, saturation=(-20, 20)):
        return self.hsv(saturation=saturation)


########################################################################################################################
class RandomDistribution:
    def __init__(self, random_f, name, **kwargs):
        self._f = random_f
        self._name = name
        self._kwargs = kwargs

    def __call__(self, rng, shape=None):
        return self._f(rng=rng, shape=shape, **self._kwargs)
    
    def __repr__(self):
        return f"RandomDistribution.{self._name}(**{self._kwargs})"

    def __getattr__(self, item):
        if item in self._kwargs:
            return self._kwargs[item]

    def __setattr__(self, key, value):
        if not key.startswith('_') and key in self._kwargs:
            self._kwargs[key] = value
        else:
            super(RandomDistribution, self).__setattr__(key, value)
        
    def __eq__(self, other):
        return self._name == other._name and self._kwargs==other._kwargs
    
    def __neq__(self, other):
        return self._name != other._name or self._kwargs!=other._kwargs

    @staticmethod
    def auto(info, symetric=False):
        """
        Generate a RandomDistribution according to the value of an argument
        :rtype: RandomDistribution
        """
        if isinstance(info, str):
            if '±' in info:
                mean, std = info.split('±')
                return RandomDistribution.normal(float(mean), float(std))
        elif isinstance(info, (tuple, list)):
            if len(info) == 2:
                return RandomDistribution.uniform(*info)
            elif len(info) == 1:
                return RandomDistribution.uniform(low=-info[0], high=+info[0])
        elif isinstance(info, (float, int)):
            if symetric:
                return RandomDistribution.uniform(low=-info, high=info)
            else:
                return RandomDistribution.uniform(high=info)
        elif isinstance(info, RandomDistribution):
            return info
        raise ValueError('Not interpretable random distribution: %s.' % repr(info))

    @staticmethod
    def discrete_uniform(values):
        if isinstance(values, (list, tuple, set, np.ndarray)):
            values = np.array(values)
            def f(rng: np.random.RandomState, shape, distribution):
                return distribution[rng.randint(low=0, high=len(distribution), size=shape)]
            return RandomDistribution(f, distribution=values)
        elif isinstance(values, int):
            def f(rng: np.random.RandomState, shape, distribution):
                return rng.randint(low=0, high=distribution, size=shape)

            return RandomDistribution(f, 'discrete_uniform', distribution=values)

    @staticmethod
    def uniform(high=1, low=0):
        if high < low:
            low, high = high, low

        def f(rng: np.random.RandomState, shape, low, high):
            return rng.uniform(low=low, high=high, size=shape)
        return RandomDistribution(f, 'uniform', low=low, high=high)

    @staticmethod
    def normal(mean=0, std=1):
        def f(rng: np.random.RandomState, shape, mean, std):
            return rng.normal(loc=mean, scale=std, size=shape)
        return RandomDistribution(f, 'normal', mean=mean, std=std)

    @staticmethod
    def truncated_normal(mean=0, std=1, truncate_high=1, truncate_low=None):
        if truncate_low is None:
            truncate_low = -truncate_high

        def f(rng, shape, mean, std, truncate_low, truncate_high):
            return np.clip(rng.normal(loc=mean, scale=std, size=shape), a_min=truncate_low, a_max=truncate_high)
        return RandomDistribution(f, 'truncated_normal', mean=mean, std=std, truncate_high=truncate_high, truncate_low=truncate_low)

    @staticmethod
    def binary(p=0.5):
        def f(rng: np.random.RandomState, shape, p):
            return rng.binomial(n=1, p=p, size=shape) > 0
        return RandomDistribution(f, 'binary', p=p)

    @staticmethod
    def constant(c=0):
        def f(rng, shape, c):
            return np.ones(shape=shape, dtype=type(c))*c
        return RandomDistribution(f, 'constant', c=c)

    @staticmethod
    def custom(f_dist, **kwargs):
        def f(rng, shape, **kwargs):
            return f_dist(x=rng.uniform(0, 1, size=shape), **kwargs)

        return RandomDistribution(f, 'custom: '+f_dist.__name__, **kwargs)

    @staticmethod
    def integers(low, high=None, dtype='i'):
        def f(rng, shape, low, high, dtype):
            return rng.integers(low, high=high, size=shape, dtype=dtype)
        return RandomDistribution(f, 'randint', low=low, high=high, dtype=dtype)


_RD = RandomDistribution


########################################################################################################################
def bind_args(f, args=(), kwargs=None):
    bind = bind_args_partial(f, args, kwargs)
    missing_args = set(not_optional_args(f)).intersection(bind.keys())
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