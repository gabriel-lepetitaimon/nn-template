import torch
import numpy as np


def crop_pad(img, size, center=None):
    if torch.is_tensor(img):
        H, W = img.shape[-2:]
    else:
        H, W = img.shape[:2]
    h, w = size
    if center is None:
        y, x = H // 2, W // 2
    else:
        y, x = center
    half_w = w // 2
    half_h = h // 2

    y0 = int(max(0, half_h - y))
    y1 = int(max(0, y - half_h))
    h = int(min(h, H - y1) - y0)

    x0 = int(max(0, half_w - w))
    x1 = int(max(0, x - half_w))
    w = int(min(w, W - x1) - x0)

    if torch.is_tensor(img):
        r = torch.zeros(tuple(img.shape[:-2]) + size, dtype=img.dtype, device=img.device)
        r[..., y0:y0 + h, x0:x0 + w] = img[..., y1:y1 + h, x1:x1 + w]
    else:
        r = np.zeros_like(img, shape=size + img.shape[2:])
        r[y0:y0 + h, x0:x0 + w] = img[y1:y1 + h, x1:x1 + w]

    return r


def prepare_lut(lut_map, source_dtype=None, axis=None, sampling=None, default=None, keep_dims=True):
    assert isinstance(lut_map, dict) and len(lut_map)

    import numpy as np
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
                source = str2color(source, uint8=str(source_dtype) == 'uint8')
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
            dest = str2color(dest, uint8=str(source_dtype) == 'uint8')
        dest = np.array(dest)
        if dest.ndim == 0:
            dest = dest.reshape((1,))
        if dest_shape is None:
            dest_shape = dest.shape
        elif dest_shape != dest.shape:
            raise ValueError('Invalid destination values: %s (shape should be %s)' % (repr(source), dest_shape))

        if source != 'default':
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

    def f_lut(array):
        if len(axis) > 1 and axis != np.arange(len(axis)):
            array = np.moveaxis(array, source=axis, destination=np.arange(len(axis)))
        elif add_empty_axis:
            array = array.reshape((1,) + array.shape)

        # if 'int' not in str(array.dtype):
        #     log.warn('Array passed to apply_lut was converted to int32. Numeric precision may have been lost.')

        # Read array shape
        a_source_shape = array.shape[:n_axis]
        map_shape = array.shape[n_axis:]
        map_size = int(np.prod(map_shape))

        # Check source shape
        if a_source_shape != source_shape:
            raise ValueError('Invalid dimensions on axis: %s. (expected: %s, received: %s)'
                             % (str(axis), str(source_shape), str(a_source_shape)))

        # Prepare table
        array = np.moveaxis(array.reshape(source_shape + (map_size,)), -1, 0).reshape((map_size, source_size))

        if isinstance(sampling, str):
            id_mapped = None
        else:
            if sampling != 1:
                array = (array / sampling).astype(np.int32)
            id_mapped = np.logical_not(np.any(np.logical_or(np.logical_or(array > maxs, array < mins), np.isnan(array)), axis=1))
            array = np.sum((array - mins) * stride, axis=1).astype(np.uint32)

        # Map values
        if isinstance(lut_sources, str):    # and lut_sources == 'nearest':
            a = np.sum((array[np.newaxis, :, :] - sources[:, np.newaxis, :]) * 2, axis=-1)
            a = np.argmin(a, axis=0) + 1
        elif id_mapped is not None:
            a = np.zeros(shape=(map_size,), dtype=np.uint32)
            if lut_sources is not None:
                a[id_mapped] = lut_sources[array[id_mapped]]
            else:
                a[id_mapped] = array[id_mapped] + 1
        else:
            if lut_sources is not None:
                a = lut_sources[array]
            else:
                a = array + 1
        array = lut_dests[a]

        del a
        del id_mapped

        # Reshape
        array = array.reshape(map_shape + dest_shape)

        array = np.moveaxis(array, np.arange(len(map_shape), array.ndim),
                            np.arange(dest_axis, dest_axis + len(dest_shape)) if len(dest_shape) != len(axis) else axis)
        if not keep_dims and dest_shape == (1,):
            array = array.reshape(map_shape)

        return array

    f_lut.sources = sources
    f_lut.lut_sources = lut_sources
    if isinstance(lut_sources, np.ndarray):
        f_lut.mins = mins
        f_lut.maxs = maxs
        f_lut.stride = stride
    f_lut.lut_dests = lut_dests
    f_lut.sampling = sampling
    f_lut.source_dtype = source_dtype
    f_lut.keep_dims = keep_dims
    return f_lut


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
