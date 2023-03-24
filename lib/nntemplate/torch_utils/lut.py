import numpy as np


def prepare_lut(lut_map, source_dtype=None, axis=None, sampling=None, default=None, keep_dims=True, bgr=True):
    assert isinstance(lut_map, dict) and len(lut_map)

    if default is not None:
        lut_map['default'] = default
        default = None

    # Prepare map
    source_list = []
    dest_list = []
    source_shape = None
    dest_shape = None

    add_empty_axis = False
    source, dest = None, None
    for source, dest in lut_map.items():
        if source != 'default':
            if isinstance(source, str):
                source = str2color(source, bgr=bgr, uint8=str(source_dtype) == 'uint8')
            source = np.array(source)
            if source.ndim == 0:
                source = source.reshape((1,))
                add_empty_axis = True
            if source_shape is None:
                source_shape = source.shape
            elif source_shape != source.shape:
                raise ValueError('Invalid source values: %s (shape should be %s)' % (repr(source), source_shape))
            source_list.append(source)

        if isinstance(dest, str):
            dest = str2color(dest, bgr=bgr, uint8=str(source_dtype) == 'uint8')
        dest = np.array(dest)
        if dest.ndim == 0:
            dest = dest.reshape((1,))
        if dest_shape is None:
            dest_shape = dest.shape
        elif dest_shape != dest.shape:
            raise ValueError('Invalid destination values: %s (shape should be %s)' % (repr(source), dest_shape))

        if not isinstance(source, str) or source != 'default':
            dest_list.append(dest)
        else:
            default = dest

    if axis:
        if isinstance(axis, int):
            axis = np.array([axis])
        elif isinstance(axis, (list, tuple, np.ndarray)):
            axis = np.array(axis)
        else:
            raise ValueError('Invalid axis parameter: %s (should be one or a list of axis)' % repr(axis))
    elif axis is None:
        axis = np.arange(len(source_shape))

    # Read shape
    n_axis = len(axis)
    source_size = int(np.prod(source_shape))
    dest_axis = sorted(axis)[0]

    # Prepare lut table
    sources = []
    lut_dests = [if_none(default, np.zeros_like(dest))]
    for s, d in zip(source_list, dest_list):
        source = np.array(s).flatten()
        dest = np.array(d)
        if dest.shape:
            dest = dest.flatten()
        sources.append(source)
        lut_dests.append(dest)

    sources = np.array(sources).astype(dtype=source_dtype)
    lut_dests = np.array(lut_dests)

    mins = sources.min(axis=0)
    maxs = sources.max(axis=0)

    if sampling is None:
        if 'float' in str(sources.dtype) and mins.min() >= 0 and maxs.max() <= 1:
            sampling = 1 / 255
    elif sampling == 'gcd':
        sampling = np.zeros(sources.shape[1:], dtype=np.float)
        for i in range(sources.shape[0]):
            sampling[i] = 1 / np.gcd.reduce(sources[i]) / 2
    if not sampling:
        sampling = 1

    if not isinstance(sampling, str):
        sources = (sources / sampling).astype(np.int32)
        mins = sources.min(axis=0)
        maxs = sources.max(axis=0)
        stride = np.cumprod([1] + list((maxs - mins + 1)[1:][::-1]), dtype=np.uint32)[::-1]

        flatten_sources = np.sum((sources - mins) * stride, dtype=np.uint32, axis=1)
        id_sorted = flatten_sources.argsort()
        flatten_sources = flatten_sources[id_sorted]
        lut_dests[1:] = lut_dests[1:][id_sorted]

        if np.all(flatten_sources == np.arange(len(flatten_sources))):
            lut_sources = None
        else:
            lut_sources = np.zeros((int(np.prod(maxs - mins + 1)),), dtype=np.uint32)
            for s_id, s in enumerate(flatten_sources):
                lut_sources[s] = s_id + 1
    else:
        lut_sources = 'nearest'
        stride = 1
        mins = 0

    return LutMap(axis, add_empty_axis, n_axis, source_shape, source_size, sampling, maxs, mins, stride,
                  lut_sources, sources, lut_dests, dest_shape, dest_axis, keep_dims)


class LutMap:
    def __init__(self, axis, add_empty_axis, n_axis, source_shape, source_size, sampling, maxs, mins, stride,
                 lut_sources, sources, lut_dests, dest_shape, dest_axis, keep_dims):
        self.axis = axis
        self.add_empty_axis = add_empty_axis
        self.n_axis = n_axis
        self.source_shape = source_shape
        self.source_size = source_size
        self.sampling = sampling
        self.maxs = maxs
        self.mins = mins
        self.stride = stride
        self.lut_sources = lut_sources
        self.sources = sources
        self.lut_dests = lut_dests
        self.dest_shape = dest_shape
        self.dest_axis = dest_axis
        self.keep_dims = keep_dims

    def __call__(self, array):
        if len(self.axis) > 1 and self.axis != np.arange(len(self.axis)):
            array = np.moveaxis(array, source=self.axis, destination=np.arange(len(axis)))
        elif self.add_empty_axis:
            array = array.reshape((1,) + array.shape)

        # if 'int' not in str(array.dtype):
        #     log.warn('Array passed to apply_lut was converted to int32. Numeric precision may have been lost.')

        # Read array shape
        a_source_shape = array.shape[:self.n_axis]
        map_shape = array.shape[self.n_axis:]
        map_size = int(np.prod(map_shape))

        # Check source shape
        if a_source_shape != self.source_shape:
            raise ValueError('Invalid dimensions on axis: %s. (expected: %s, received: %s)'
                             % (str(self.axis), str(self.source_shape), str(a_source_shape)))

        # Prepare table
        array = np.moveaxis(array.reshape(self.source_shape + (map_size,)), -1, 0).reshape((map_size, self.source_size))

        if isinstance(self.sampling, str):
            id_mapped = None
        else:
            if self.sampling != 1:
                array = (array / self.sampling).astype(np.int32)
            id_mapped = np.logical_not(
                np.any(np.logical_or(np.logical_or(array > self.maxs, array < self.mins), np.isnan(array)), axis=1))
            array = np.sum((array - self.mins) * self.stride, axis=1).astype(np.uint32)

        # Map values
        if isinstance(self.lut_sources, str):  # and lut_sources == 'nearest':
            a = np.sum((array[np.newaxis, :, :] - self.sources[:, np.newaxis, :]) * 2, axis=-1)
            a = np.argmin(a, axis=0) + 1
        elif id_mapped is not None:
            a = np.zeros(shape=(map_size,), dtype=np.uint32)
            if self.lut_sources is not None:
                a[id_mapped] = self.lut_sources[array[id_mapped]]
            else:
                a[id_mapped] = array[id_mapped] + 1
        else:
            if self.lut_sources is not None:
                a = self.lut_sources[array]
            else:
                a = array + 1
        array = self.lut_dests[a]

        del a
        del id_mapped

        # Reshape
        array = array.reshape(map_shape + self.dest_shape)

        array = np.moveaxis(array, np.arange(len(map_shape), array.ndim),
                            np.arange(self.dest_axis, self.dest_axis + len(self.dest_shape)) if len(self.dest_shape) != len(
                                self.axis) else self.axis)
        if not self.keep_dims and self.dest_shape == (1,):
            array = array.reshape(map_shape)

        return array


def apply_lut(array, lut_map, axis=None, sampling=None, default=None):
    # import numpy as np
    #
    # a = array
    # if axis:
    #     if isinstance(axis, int):
    #         axis = np.array([axis])
    #     elif isinstance(axis, (list, tuple, np.ndarray)):
    #         axis = np.array(axis)
    #         a = np.moveaxis(a, source=axis, destination=np.arange(len(axis)))
    #     else:
    #         raise ValueError('Invalid axis parameter: %s (should be one or a list of axis)' % repr(axis))
    # elif axis is None:
    #     axis = np.arange(np.array(next(iter(map.keys()))).ndim)
    #     if len(axis) == 0:
    #         axis = None
    #         a = array.reshape((1,) + a.shape)
    #
    # n_axis = len(axis) if axis else 1
    # source_shape = a.shape[:n_axis]
    # source_size = int(np.prod(source_shape))
    # map_shape = a.shape[n_axis:]
    # map_size = int(np.prod(map_shape))
    #
    # a = a.reshape((source_size, map_size))
    # mins = a.min(axis=-1)
    # maxs = a.max(axis=-1)
    # a_minmax = (mins, maxs)

    f_lut = prepare_lut(lut_map, source_dtype=array.dtype, axis=axis, sampling=sampling, default=default)
    return f_lut(array)


def str2color(str_color, bgr=True, uint8=True):
    import numpy as np
    if not str_color or not isinstance(str_color, str):
        return np.zeros((3,), dtype=np.uint8 if uint8 else np.float16)

    c = str_color.split('.')
    if len(c) == 1:
        c_str = c[0]
        m = 1
    else:
        c_str = c[0]
        m = float('0. ' + c[1])
        if c_str.lower() == 'black':
            m = 1 - m
    import webcolors
    try:
        c = webcolors.name_to_rgb(c_str)
    except ValueError:
        try:
            c = webcolors.hex_to_rgb(c_str)
        except ValueError:
            raise ValueError('Invalid color code: %s' % c) from None
    c = np.array(c, dtype=np.float16) * m
    if uint8:
        c = c.astype(dtype=np.uint8)
    else:
        c /= 255
    if bgr:
        c = c[::-1]
    return c


def if_none(v, o):
    return o if v is None else v
