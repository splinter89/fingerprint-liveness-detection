"""Filterbank Correlation Operation"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#          Giovani Chiachia <giovani.chiachia@gmail.com>
#
# License: BSD

__all__ = ['fbcorr4']

import numpy as np
from skimage.util.shape import view_as_windows

DEFAULT_STRIDE = 1


def fbcorr4(arr_in, arr_fb, stride=DEFAULT_STRIDE, arr_out=None):

    """4D Filterbank Correlation
    XXX: docstring
    """

    assert arr_in.ndim == 4
    assert arr_fb.ndim == 4
    assert arr_fb.dtype == arr_in.dtype

    in_imgs, inh, inw, ind = arr_in.shape
    fbh, fbw, fbd, fbn = arr_fb.shape

    f_size = fbh * fbw * fbd

    assert fbn > 1
    assert fbh <= inh
    assert fbw <= inw
    assert fbd == ind

    if arr_out is not None:
        assert arr_out.dtype == arr_in.dtype
        assert arr_out.shape == (in_imgs,
                                 1 + (inh - fbh) / stride,
                                 1 + (inw - fbw) / stride,
                                 fbn)

    # -- reshape arr_in
    arr_inr = view_as_windows(arr_in, (1, fbh, fbw, fbd))[::stride, ::stride]

    n_imgs, outh, outw = arr_inr.shape[:3]
    assert n_imgs == in_imgs

    arr_inrm = arr_inr.reshape(n_imgs * outh * outw, f_size)

    # -- reshape arr_fb
    arr_fbm = arr_fb.reshape((f_size, fbn))

    # -- correlate!
    arr_out = np.dot(arr_inrm, arr_fbm)
    arr_out = arr_out.reshape(n_imgs, outh, outw, -1)

    return arr_out
