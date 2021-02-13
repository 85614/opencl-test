
import numbers
import sys
import os
import math
import random
import itertools
import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.base import py_str, MXNetError, _as_list
from mxnet.test_utils import set_default_context, assert_almost_equal, assert_allclose, check_numeric_gradient, check_symbolic_backward
from mxnet.symbol import Symbol
from pickle import load, dump

def default_context():
    return mx.cpu()


def test_op_roi_align():
    T = np.float32

    def assert_same_dtype(dtype_a, dtype_b):
        '''
        Assert whether the two data type are the same
        Parameters
        ----------
        dtype_a, dtype_b: type
            Input data types to compare
        '''
        assert dtype_a == dtype_b,\
            TypeError('Unmatched data types: %s vs %s' % (dtype_a, dtype_b))

    def bilinear_interpolate(bottom, height, width, y, x):
        if y < -1.0 or y > height or x < -1.0 or x > width:
            return T(0.0), []
        x = T(max(0.0, x))
        y = T(max(0.0, y))
        x_low = int(x)
        y_low = int(y)
        if x_low >= width - 1:
            x_low = x_high = width - 1
            x = T(x_low)
        else:
            x_high = x_low + 1
        if y_low >= height - 1:
            y_low = y_high = height - 1
            y = T(y_low)
        else:
            y_high = y_low + 1
        ly = y - T(y_low)
        lx = x - T(x_low)
        hy = T(1.0) - ly
        hx = T(1.0) - lx
        v1 = bottom[y_low, x_low]
        v2 = bottom[y_low, x_high]
        v3 = bottom[y_high, x_low]
        v4 = bottom[y_high, x_high]
        w1 = hy * hx
        w2 = hy * lx
        w3 = ly * hx
        w4 = ly * lx
        assert_same_dtype(w1.dtype, T)
        assert_same_dtype(w2.dtype, T)
        assert_same_dtype(w3.dtype, T)
        assert_same_dtype(w4.dtype, T)
        val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
        assert_same_dtype(val.dtype, T)
        grad = [(y_low, x_low, w1), (y_low, x_high, w2),
                (y_high, x_low, w3), (y_high, x_high, w4)
                ]
        return val, grad

    def roialign_forward_backward(data, rois, pooled_size, spatial_scale, sampling_ratio,
                                  position_sensitive, dy):
        N, C, H, W = data.shape
        R = rois.shape[0]
        PH, PW = pooled_size
        assert rois.ndim == 2,\
            ValueError(
                'The ndim of rois should be 2 rather than %d' % rois.ndim)
        assert rois.shape[1] == 5,\
            ValueError(
                'The length of the axis 1 of rois should be 5 rather than %d' % rois.shape[1])
        assert_same_dtype(data.dtype, T)
        assert_same_dtype(rois.dtype, T)

        C_out = C // PH // PW if position_sensitive else C
        out = np.zeros((R, C_out, PH, PW), dtype=T)
        dx = np.zeros_like(data)
        drois = np.zeros_like(rois)

        for r in range(R):
            batch_ind = int(rois[r, 0])
            sw, sh, ew, eh = rois[r, 1:5] * T(spatial_scale)
            roi_w = T(max(ew - sw, 1.0))
            roi_h = T(max(eh - sh, 1.0))
            bin_h = roi_h / T(PH)
            bin_w = roi_w / T(PW)
            bdata = data[batch_ind]
            if sampling_ratio > 0:
                roi_bin_grid_h = roi_bin_grid_w = sampling_ratio
            else:
                roi_bin_grid_h = int(np.ceil(roi_h / T(PH)))
                roi_bin_grid_w = int(np.ceil(roi_w / T(PW)))
            count = T(roi_bin_grid_h * roi_bin_grid_w)
            for c in range(C_out):
                for ph in range(PH):
                    for pw in range(PW):
                        val = T(0.0)
                        c_in = c * PH * PW + ph * PW + pw if position_sensitive else c
                        for iy in range(roi_bin_grid_h):
                            y = sh + T(ph) * bin_h + (T(iy) + T(0.5)) * \
                                bin_h / T(roi_bin_grid_h)
                            for ix in range(roi_bin_grid_w):
                                x = sw + T(pw) * bin_w + (T(ix) + T(0.5)) * \
                                    bin_w / T(roi_bin_grid_w)
                                v, g = bilinear_interpolate(
                                    bdata[c_in], H, W, y, x)
                                assert_same_dtype(v.dtype, T)
                                val += v
                                # compute grad
                                for qy, qx, qw in g:
                                    assert_same_dtype(qw.dtype, T)
                                    dx[batch_ind, c_in, qy, qx] += dy[r,
                                                                      c, ph, pw] * qw / count
                        out[r, c, ph, pw] = val / count
        assert_same_dtype(out.dtype, T)
        return out, [dx, drois]

    def test_roi_align_value(sampling_ratio=0, position_sensitive=False):
        ctx = default_context()
        dtype = np.float32
        dlen = 224
        N, C, H, W = 5, 3, 16, 16
        R = 7
        pooled_size = (3, 4)
        C = C * pooled_size[0] * pooled_size[1] if position_sensitive else C
        spatial_scale = H * 1.0 / dlen
        data = mx.nd.array(
            np.arange(N * C * W * H).reshape((N, C, H, W)), ctx=ctx, dtype=dtype)
        center_xy = mx.nd.random.uniform(0, dlen, (R, 2), ctx=ctx, dtype=dtype)
        wh = mx.nd.random.uniform(0, dlen, (R, 2), ctx=ctx, dtype=dtype)
        batch_ind = mx.nd.array(np.random.randint(0, N, size=(R, 1)), ctx=ctx)
        pos = mx.nd.concat(center_xy - wh / 2, center_xy + wh / 2, dim=1)
        rois = mx.nd.concat(batch_ind, pos, dim=1)

        data.attach_grad()
        rois.attach_grad()
        with mx.autograd.record():
            output = mx.nd.contrib.ROIAlign(data, rois, pooled_size=pooled_size,
                                            spatial_scale=spatial_scale, sample_ratio=sampling_ratio,
                                            position_sensitive=position_sensitive)
        C_out = C // pooled_size[0] // pooled_size[1] if position_sensitive else C
        dy = mx.nd.random.uniform(-1, 1, (R, C_out) +
                                  pooled_size, ctx=ctx, dtype=dtype)
        output.backward(dy)
        real_output, [dx, drois] = roialign_forward_backward(data.asnumpy(), rois.asnumpy(), pooled_size,
                                                             spatial_scale, sampling_ratio,
                                                             position_sensitive, dy.asnumpy())

        assert_almost_equal(output, real_output, atol=1e-3)
        assert_almost_equal(data.grad, dx, atol=1e-3)
        assert_almost_equal(rois.grad, drois, atol=1e-3)

    # modified from test_roipooling()
    def test_roi_align_autograd(sampling_ratio=0):
        ctx = default_context()
        data = mx.symbol.Variable(name='data')
        rois = mx.symbol.Variable(name='rois')
        test = mx.symbol.contrib.ROIAlign(data=data, rois=rois, pooled_size=(4, 4), spatial_scale=1,
                                          sample_ratio=sampling_ratio)

        x1 = np.random.rand(4, 1, 12, 12).astype('float64')
        x2 = np.array([[0, 1.1, 1.1, 6.2, 6.2], [2, 6.1, 2.1, 8.2, 11.2],
                       [1, 3.1, 1.1, 5.2, 10.2]], dtype='float64')

        check_numeric_gradient(sym=test, location=[x1, x2],
                               grad_nodes={'data': 'write', 'rois': 'null'},
                               numeric_eps=1e-4, rtol=1e-1, atol=1e-4, ctx=ctx)
        check_numeric_gradient(sym=test, location=[x1, x2],
                               grad_nodes={'data': 'add', 'rois': 'null'},
                               numeric_eps=1e-4, rtol=1e-1, atol=1e-4, ctx=ctx)

    test_roi_align_value()
    print('correct')
    test_roi_align_value(sampling_ratio=2)
    print('correct')
    test_roi_align_value(position_sensitive=True)
    print('correct')
    test_roi_align_autograd()
    print('correct')



def my_test():        
    test_op_roi_align()



if __name__ == '__main__':
    my_test()
    
        
    
    