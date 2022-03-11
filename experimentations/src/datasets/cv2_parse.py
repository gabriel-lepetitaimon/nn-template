import cv2


class ParseDict:
    def __init__(self, name, dict):
        self._name = name
        self._dict = dict

    def __call__(self, v, default=None):
        v = v.lower()
        if v not in self._dict:
            return default
        return self._dict[v]

    def __getitem__(self, v):
        v = v.lower()
        if v not in self._dict:
            raise ValueError(f'Invalid {self._name}: "{v}". \n'
                             f'Valid values are {list(self._dict.keys())}.')
        return self._dict[v]

    def __iter__(self):
        return iter(self._dict)

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def values(self):
        return self._dict.values()

    def __len__(self):
        return len(self._dict)


INTERPOLATIONS = ParseDict('interpolation', {
    'linear': cv2.INTER_LINEAR,
    'nearest': cv2.INTER_NEAREST,
    'area': cv2.INTER_AREA,
    'cubic': cv2.INTER_CUBIC,
    'max': cv2.INTER_MAX,
    'bits': cv2.INTER_BITS,
})


BORDER_MODES = ParseDict('border mode', {
    'constant': cv2.BORDER_CONSTANT,
    'replicate': cv2.BORDER_REPLICATE,
    'wrap': cv2.BORDER_WRAP,
    'reflect': cv2.BORDER_REFLECT,
    'reflect101': cv2.BORDER_REFLECT101,
})