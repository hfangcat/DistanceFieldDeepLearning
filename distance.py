""""
'Distance Transform of Sampled Functions'
by Felzenszwalb and Huttenlocher.
"""

from numba import jit
import numpy as np
import snowy

INF = 1e20


def generate_sdf(image: np.ndarray, wrapx=False, wrapy=False):
    a = generate_udf(image, wrapx, wrapy)
    b = generate_udf(image == 0.0, wrapx, wrapy)
    return a - b


def generate_udf(image: np.ndarray, wrapx=False, wrapy=False):
    assert image.dtype == 'bool', 'Pixel values must be boolean'
    assert len(image.shape) == 3, 'Shape is not rows x cols x channels'
    assert image.shape[2] == 1, 'Image must be grayscale'
    return _generate_edt(image, wrapx, wrapy)


def _generate_edt(image, wrapx, wrapy):
    # 3d -> 2d
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = np.reshape(image, image.shape[: 2])
    result = np.where(image, 0.0, INF)
    _generate_udt(result, wrapx, wrapy)
    # 2d -> 3d
    if len(result.shape) == 2:
        result = np.reshape(result, result.shape + (1, ))
    return np.sqrt(result)


def _generate_udt(result, wrapx, wrapy):
    scratch = result
    if wrapx:
        scratch = np.hstack([scratch, scratch, scratch])
    if wrapy:
        scratch = np.vstack([scratch, scratch, scratch])

    height, width = scratch.shape
    capacity = max(height, width)
    i = np.empty(scratch.shape, dtype='u2')
    j = np.empty(scratch.shape, dtype='u2')

    d = np.zeros([capacity])
    z = np.zeros([capacity + 1])
    v = np.zeros([capacity], dtype='u2')

    _generate_udt_native(width, height, d, z, v, i, j, scratch)

    x0, x1 = width // 3, 2 * width // 3
    y0, y1 = height // 3, 2 * height // 3
    if wrapx:
        scratch = scratch[:, x0:x1]
    if wrapy:
        scratch = scratch[y0:y1, :]
    if wrapx or wrapy:
        np.copyto(result, scratch)

    return i, j


@jit(nopython=True, fastmath=True, cache=True)
def _generate_udt_native(width, height, d, z, v, i, j, result):
    for x in range(width):
        f = result[:, x]
        edt(f, d, z, v, j[:, x], height)
        result[:, x] = d[: height]
    for y in range(height):
        f = result[y, :]
        edt(f, d, z, v, i[y, :], width)
        result[y, :] = d[: width]


@jit(nopython=True, fastmath=True, cache=True)
def edt(f, d, z, v, i, n):
    #   Find the lower envelope of a sequence of parabolas.
    #   f...source data (returns the Y of the parabola vertex at X)
    #   d...destination data (final distance values are written here)
    #   z...temporary used to store X coords of parabola intersections
    #   v...temporary used to store X coords of parabola vertices
    #   i...resulting X coords of parabola vertices
    #   n...number of pixels in "f" to process
    k: int = 0
    v[0] = 0
    z[0] = -INF
    z[1] = INF

    for q in range(1, n):
        p = v[k]
        s = ((f[q] + q * q) - (f[p] + p * p)) / (2.0 * q - 2.0 * p)
        while s <= z[k]:
            k = k - 1
            p = v[k]
            s = ((f[q] + q * q) - (f[p] + p * p)) / (2.0 * q - 2.0 * p)

        k = k + 1
        v[k] = q
        z[k] = s
        z[k + 1] = +INF

    k = 0
    for q in range(n):
        while z[k + 1] < float(q):
            k = k + 1
        dx = q - v[k]
        d[q] = dx * dx + f[v[k]]
        i[q] = v[k]


image = snowy.load("test_mask_preprocessed/pre_processed_mask_61.png")
image = image[:, :, 1]
image = image[:, :, np.newaxis]
sdf = generate_sdf(image != 0.0)
