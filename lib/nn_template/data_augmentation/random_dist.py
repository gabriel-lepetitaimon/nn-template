import numpy as np
from ..config import Cfg


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


class RandDistAttr(Cfg.Attr):
    def __init__(self, default='__undefined__', symetric=False):
        super(RandDistAttr, self).__init__(default=default)
        self.symetric = symetric

    def check_value(self, value):
        value = super(RandDistAttr, self).check_value(value)
        try:
            return RandomDistribution.auto(value, symetric=self.symetric)
        except ValueError as e:
            raise Cfg.InvalidAttr(str(e))
