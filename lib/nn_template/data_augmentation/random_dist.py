import numpy as np
from ..config.cfg_object import CfgAttr, InvalidAttr, CfgDict


class RandomDistribution:
    def __init__(self, random_f, name, **kwargs):
        self._f = random_f
        self._name = name
        self._kwargs = kwargs

    def __call__(self, rng, shape=None):
        return self._f(rng=rng, shape=shape, **self._kwargs)

    def __repr__(self):
        return f"RandomDistribution.{self._name}({', '.join([str(k)+'='+repr(v) for k,v in self._kwargs.items()])})"

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
            try:
                if '±' in info:
                    mean, std = info.split('±')
                    if mean == '':
                        mean = 0
                    return RandomDistribution.normal(interpret_float(mean), interpret_float(std))
                else:
                    info = interpret_float(info)
            except TypeError:
                pass
        if isinstance(info, (tuple, list)):
            if len(info) == 2:
                return RandomDistribution.uniform(*info)
            elif len(info) == 1:
                return RandomDistribution.uniform(low=-info[0], high=+info[0])
        elif isinstance(info, (float, int)) and info is not True and info is not False:
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
    def normal(mean: float = 0, std: float = 1):
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


class RandDistAttr(CfgAttr):
    def __init__(self, default='__undefined__', symetric=False):
        self.symetric = symetric
        super(RandDistAttr, self).__init__(default=default)

    def _check_value(self, value, cfg_dict: CfgDict | None = None):
        try:
            return RandomDistribution.auto(value, symetric=self.symetric)
        except ValueError as e:
            raise InvalidAttr(str(e))


def interpret_float(value) -> float:
    if isinstance(value, str):
        value = value.strip()
        if value.endswith('%'):
            value = float(value[:-1])/100
        elif value.endswith('‰'):
            value = float(value[:-1])/100
        elif value.endswith(tuple('TGMkmµn')):
            value = float(value[:-1])*{
                'T': 1e12, 'G': 1e9, 'M': 1e6, 'k': 1e3, 'm': 1e-3, 'µ': 1e-6, 'n': 1e-9
            }[value[-1]]
    return float(value)


def interpret_int(value) -> int:
    if isinstance(value, str):
        if value.endswith(tuple('TGMk')):
            value = float(value[:-1])*{
                'T': 1e12, 'G': 1e9, 'M': 1e6, 'k': 1e3
            }[value[-1]]
    return int(value)
