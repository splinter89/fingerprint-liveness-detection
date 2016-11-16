"""
Test suite for ```extractor```
"""

import numpy as np
from numpy.testing import assert_allclose
import cnnrandom as cnnr

RTOL = 1e-5
ATOL = 1e-6


def test_no_model():

    in_shape = (200, 200)

    model = []
    extractor = cnnr.BatchExtractor(in_shape, model)

    assert extractor.n_layers == 0
    assert extractor.shape_stride_by_layer == []
    assert extractor.filterbanks == []


def test_ht_l3_1():

    in_shape = (200, 200)

    model = cnnr.models.fg11_ht_l3_1_description
    extractor = cnnr.BatchExtractor(in_shape, model)

    assert extractor.n_layers == 4
    assert extractor.shape_stride_by_layer == [(192, 192,  9,  9,  1,  1),
                                               (88,   88, 17, 17, 64,  2),
                                               (34,   34, 21, 21, 128, 2),
                                               (10,   10, 15, 15, 256, 2)]


def test_zero_images():

    in_shape = (200, 200)
    n_imgs = 12

    model = cnnr.models.fg11_ht_l3_1_description
    extractor = cnnr.BatchExtractor(in_shape, model)

    imgs = np.zeros((n_imgs,) + in_shape).astype(np.float32)

    feat_set = extractor.extract(imgs)

    assert feat_set.shape == (n_imgs, 10, 10, 256)
    assert feat_set.sum() == 0.


def test_random_images():

    in_shape = (200, 200)
    n_imgs = 12

    model = cnnr.models.fg11_ht_l3_1_description
    extractor = cnnr.BatchExtractor(in_shape, model)

    rng = np.random.RandomState(42)
    imgs = rng.uniform(low=0.0, high=1.0,
                       size=(n_imgs,) + in_shape).astype(np.float32)

    feat_set = extractor.extract(imgs)

    assert feat_set.shape == (n_imgs, 10, 10, 256)

    feat_set.shape = n_imgs, -1
    test_chunk_computed = feat_set[3:9, 12798:12802]

    test_chunk_expected = np.array([
        [0.0211054,  0.03008464, 0.01754513, 0.02833426],
        [0.02208067, 0.00935635, 0.01337241, 0.01570413],
        [0.009751,   0.01863629, 0.01336291, 0.01315528],
        [0.01786099, 0.01888827, 0.01754452, 0.00985618],
        [0.01402772, 0.01768549, 0.01073015, 0.01337401],
        [0.02214634, 0.0261642,  0.00918513, 0.02125477]],
        dtype=np.float32)

    assert_allclose(test_chunk_computed, test_chunk_expected,
                    rtol=RTOL, atol=ATOL)


def test_lena():

    from skimage import data
    from skimage import color
    from skimage.transform import resize

    in_shape = (200, 200)
    n_imgs = 1

    lena = resize(color.rgb2gray(data.lena()), in_shape).astype(np.float32)
    lena -= lena.min()
    lena /= lena.max()

    imgs = lena.reshape((n_imgs,)+in_shape)

    model = cnnr.models.fg11_ht_l3_1_description
    extractor = cnnr.BatchExtractor(in_shape, model)

    feat_set = extractor.extract(imgs)

    assert feat_set.shape == (n_imgs, 10, 10, 256)

    feat_set.shape = n_imgs, -1
    test_chunk_computed = feat_set[0, 12798:12802]

    test_chunk_expected = np.array(
        [0.03845372, 0.02469639, 0.01009409, 0.02500059], dtype=np.float32)

    assert_allclose(test_chunk_computed, test_chunk_expected,
                    rtol=RTOL, atol=ATOL)
