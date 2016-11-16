"""Extract Features in Mini-Batches according to the Provided Model"""

# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>
#          Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

import time
import numpy as np
import numexpr as ne
from pprint import pprint

from _lnorm import lcdnorm4
from _fbcorr import fbcorr4
from _lpool import lpool4

# --
DTYPE = np.float32
MINI_BATCH_SIZE = 35

# ----------------
# Helper functions
# ----------------


def _get_rf_shape_stride_in_interval(model,
                                     ly_ini, op_ini,
                                     ly_end, op_end,
                                     depth_ini):

    # -- P.S.: the inverval determined by ly_ini, op_ini,
    #          ly_end, and op_end is inclusive.

    # -- assert there is at least one operation in the interval
    assert len(model) >= ly_end >= ly_ini >= 0
    assert op_ini >= 0

    if ly_end == ly_ini:
        assert len(model[ly_end]) >= op_end >= op_ini

    # -- uppermost unit shape and stride size
    out_h, out_w, out_d, out_s = 1, 1, depth_ini, 1

    for ly_idx in reversed(xrange(ly_ini, ly_end + 1)):
        if ly_idx == ly_ini:
            it_op_ini = op_ini
        else:
            it_op_ini = 0
        if ly_idx == ly_end:
            it_op_end = op_end + 1
        else:
            it_op_end = len(model[ly_idx])

        for op_idx in reversed(xrange(it_op_ini, it_op_end)):

            operation = model[ly_idx][op_idx]

            if operation[0] == 'lnorm':
                nbh, nbw = operation[1]['kwargs']['inker_shape']
                s = 1
            elif operation[0] == 'fbcorr':
                nbh, nbw = operation[1]['initialize']['filter_shape']
                s = 1
                out_d = operation[1]['initialize']['n_filters']
            elif operation[0] == 'lpool':
                nbh, nbw = operation[1]['kwargs']['ker_shape']
                s = operation[1]['kwargs']['stride']

            # -- compute what was the shape before applying the preceding
            #    operation
            in_h = (out_h - 1) * s + nbh
            in_w = (out_w - 1) * s + nbw
            out_h, out_w, out_s = in_h, in_w, out_s * s

    return (out_h, out_w, out_d, out_s)


def _get_shape_stride_by_layer(model, in_shape):

    to_return = []
    tmp_shape = in_shape

    for layer_idx, layer_desc in enumerate(model):

        # -- get receptive field layer-wise
        rfh, rfw, rfd, rfs = _get_rf_shape_stride_in_interval(
            model, layer_idx, 0, layer_idx, len(layer_desc)-1,
            tmp_shape[-1])

        tmp_shape = ((tmp_shape[0] - rfh) / rfs + 1,
                     (tmp_shape[1] - rfw) / rfs + 1,
                     rfd)

        to_return += [(tmp_shape[0], tmp_shape[1], rfh, rfw, rfd, rfs)]

    return to_return


# ---------------
# Extractor class
# ---------------


class BatchExtractor(object):

    def __init__(self, in_shape, model):

        assert len(in_shape) == 2 or len(in_shape) == 3
        if len(in_shape) == 2:
            in_shape = in_shape + (1,)

        pprint(model)

        self.in_shape = in_shape
        self.model = model
        self.n_layers = len(model)

        self.shape_stride_by_layer = _get_shape_stride_by_layer(model,
                                                                in_shape)

        # -- set random filter weights
        self.filterbanks = self._setfilters()

        # -- this is the working array, that will be used throughout the object
        #    methods. Its purpose is to avoid a large memory footprint.
        self.arr_w = None

    def _setfilters(self):

        filterbanks = []
        model = self.model

        for layer_idx in xrange(len(model)):

            l_desc = model[layer_idx]
            op_name, op_params = l_desc[0]

            if op_name == 'fbcorr':

                # -- depth of the previous layer
                if layer_idx == 0:
                    n_f_in = self.in_shape[-1]
                else:
                    n_f_in = self.shape_stride_by_layer[layer_idx-1][4]

                f_init = op_params['initialize']
                f_shape = f_init['filter_shape'] + (n_f_in,)
                n_filters = f_init['n_filters']

                generate = f_init['generate']
                method_name, method_kwargs = generate
                assert method_name == 'random:uniform'

                rseed = method_kwargs.get('rseed', None)
                rng = np.random.RandomState(rseed)

                fb_shape = (n_filters,) + f_shape

                fb = rng.uniform(low=-1.0, high=1.0, size=fb_shape)

                # -- zero-mean, unit-l2norm
                for f_idx in xrange(n_filters):

                    filt = fb[f_idx]
                    filt -= filt.mean()
                    filt_norm = np.linalg.norm(filt)
                    assert filt_norm != 0
                    filt /= filt_norm
                    fb[f_idx] = filt

                fb = np.ascontiguousarray(np.rollaxis(fb, 0, 4)).astype(DTYPE)
                filterbanks += [fb.copy()]
            else:
                filterbanks += [[]]

        assert len(filterbanks) == len(model)

        return filterbanks

    def extract(self, arr_in):

        input_shape = arr_in.shape
        n_imgs = input_shape[0]

        assert n_imgs > 0
        assert len(input_shape) == 3 or len(input_shape) == 4

        if len(input_shape) == 3:
            arr_in.shape = arr_in.shape + (1,)

        assert arr_in.shape[1:] == self.in_shape

        model = self.model

        # assert that the model is ready to be used
        assert len(self.filterbanks) == self.n_layers

        # -- get mini-batch index intervals
        # -- determine if and how arr_in is partitioned
        mb_intervals = range(0, n_imgs, MINI_BATCH_SIZE) + [n_imgs]
        mb_intervals = [(mb_intervals[i-1], mb_intervals[i])
                        for i in xrange(1, len(mb_intervals))]
        n_mini_batches = len(mb_intervals)

        for mb_idx, (mb_init, mb_end) in enumerate(mb_intervals):

            t1 = time.time()

            if n_mini_batches == 1:
                self.arr_w = arr_in
            else:
                if mb_idx == 0:
                    # -- initialize arr_out
                    [l_h, l_w, _, _, l_d, _] = \
                        self.shape_stride_by_layer[self.n_layers-1]

                    arr_out = np.empty((n_imgs, l_h, l_w, l_d), dtype=DTYPE)

                self.arr_w = arr_in[mb_init:mb_end]
                self.arr_w.shape = (mb_end-mb_init,) + self.arr_w.shape[1:]

            for layer_idx, l_desc in enumerate(model):
                self._process_layer(layer_idx, l_desc)

            if n_mini_batches == 1:
                arr_out = self.arr_w
            else:
                arr_out[mb_init:mb_end] = self.arr_w

            t_elapsed = time.time() - t1
            print 'Mini-batch %d out of %d processed in %g seconds...' % (
                  mb_idx + 1, n_mini_batches, t_elapsed)

        self.arr_w = None
        return arr_out

    # -- transform self.arr_w according to the model layer.
    def _process_layer(self, layer_idx, l_desc):

        if layer_idx == 0:
            l_h, l_w, l_d = self.in_shape
        else:
            [l_h, l_w, _, _, l_d, _] = self.shape_stride_by_layer[layer_idx-1]

        # -- assert layer input shape
        assert self.arr_w.shape[1:] == (l_h, l_w, l_d)

        for op_idx, (op_name, op_params) in enumerate(l_desc):
            kwargs = op_params['kwargs']

            if op_name == 'fbcorr':
                fb = self.filterbanks[layer_idx]
            else:
                fb = None

            self.arr_w = self._process_one_op(op_name, kwargs, self.arr_w, fb)

        return

    def _process_one_op(self, op_name, kwargs, arr_in, fb=None):

        if op_name == 'lnorm':

            inker_shape = kwargs['inker_shape']
            outker_shape = kwargs['outker_shape']
            remove_mean = kwargs['remove_mean']
            stretch = kwargs['stretch']
            threshold = kwargs['threshold']

            # PLoS09 / FG11 constraints:
            assert inker_shape == outker_shape

            tmp_out = lcdnorm4(arr_in, inker_shape,
                               contrast=remove_mean,
                               stretch=stretch,
                               threshold=threshold)

        elif op_name == 'fbcorr':

            assert fb is not None

            max_out = kwargs['max_out']
            min_out = kwargs['min_out']

            # -- filter
            assert arr_in.dtype == np.float32

            tmp_out = fbcorr4(arr_in, fb)

            # -- activation
            min_out = -np.inf if min_out is None else min_out
            max_out = +np.inf if max_out is None else max_out
            # insure that the type is right before calling numexpr
            min_out = np.array([min_out], dtype=arr_in.dtype)
            max_out = np.array([max_out], dtype=arr_in.dtype)
            # call numexpr
            tmp_out = ne.evaluate('where(tmp_out < min_out, min_out, tmp_out)')
            tmp_out = ne.evaluate('where(tmp_out > max_out, max_out, tmp_out)')
            assert tmp_out.dtype == arr_in.dtype

        elif op_name == 'lpool':

            ker_shape = kwargs['ker_shape']
            order = kwargs['order']
            stride = kwargs['stride']

            tmp_out = lpool4(arr_in, ker_shape, order=order, stride=stride)

        else:
            raise ValueError("operation '%s' not understood" % op_name)

        assert tmp_out.dtype == arr_in.dtype
        assert tmp_out.dtype == np.float32

        return tmp_out
