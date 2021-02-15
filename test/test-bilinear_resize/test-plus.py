
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
from mxnet.test_utils import set_default_context, assert_almost_equal, assert_allclose, check_numeric_gradient, check_symbolic_backward, check_symbolic_forward
from mxnet.symbol import Symbol
from numpy.testing import assert_allclose, assert_array_equal
from pickle import load, dump

def default_context():
    return mx.cpu()


def with_seed(*args, **kwargs):
  def decorator(func):
    return func
  return decorator


@with_seed()
def test_bilinear_resize_op():
    def py_bilinear_resize(x, outputHeight, outputWidth):
        batch, channel, inputHeight, inputWidth = x.shape
        if outputHeight == inputHeight and outputWidth == inputWidth:
            return x
        y = np.empty([batch, channel, outputHeight, outputWidth])
        rheight = 1.0 * (inputHeight - 1) / (outputHeight - 1) if outputHeight > 1 else 0.0
        rwidth = 1.0 * (inputWidth - 1) / (outputWidth - 1) if outputWidth > 1 else 0.0
        for h2 in range(outputHeight):
            h1r = 1.0 * h2 * rheight
            h1 = int(np.floor(h1r))
            h1lambda = h1r - h1
            h1p = 1 if h1 < (inputHeight - 1) else 0
            for w2 in range(outputWidth):
                w1r = 1.0 * w2 * rwidth
                w1 = int(np.floor(w1r))
                w1lambda = w1r - w1
                w1p = 1 if w1 < (inputWidth - 1) else 0
                for b in range(batch):
                    for c in range(channel):
                        y[b][c][h2][w2] = (1-h1lambda)*((1-w1lambda)*x[b][c][h1][w1] + \
                            w1lambda*x[b][c][h1][w1+w1p]) + \
                            h1lambda*((1-w1lambda)*x[b][c][h1+h1p][w1] + \
                            w1lambda*x[b][c][h1+h1p][w1+w1p])
        return y
    def py_bilinear_resize_backward(x, incoming_grads, mode='size'):
        data1 = np.zeros_like(x)
        data2 = incoming_grads
        batchsize = data1.shape[0]
        channels = data1.shape[1]
        height1 = data1.shape[2]
        width1 = data1.shape[3]
        height2 = data2.shape[2]
        width2 = data2.shape[3]
        rheight = float(height1 - 1) / (height2 - 1) if (height2 > 1) else 0
        rwidth = float(width1 - 1) / (width2 - 1) if (width2 > 1) else 0
        # special case: just copy
        if height1 == height2 and width1 == width2:
            data1 += data2
            return [data1]
        for h2 in range(0, height2):
            for w2 in range(0, width2):
                h1r = rheight * h2
                h1 = int(h1r)
                h1p = 1 if (h1 < height1 - 1) else 0
                h1lambda = h1r - h1
                h0lambda = 1 - h1lambda
                #
                w1r = rwidth * w2
                w1 = int(w1r)
                w1p = 1 if (w1 < width1 - 1) else 0
                w1lambda = w1r - w1
                w0lambda = 1 - w1lambda
                #
                for n in range(0, batchsize):
                    for c in range(0, channels):
                        d2val = data2[n][c][h2][w2]
                        data1[n][c][h1][w1] += h0lambda * w0lambda * d2val
                        data1[n][c][h1][w1 + w1p] += h0lambda * w1lambda * d2val
                        data1[n][c][h1 + h1p][w1] += h1lambda * w0lambda * d2val
                        data1[n][c][h1 + h1p][w1 + w1p] += h1lambda * w1lambda * d2val
        if mode == 'like':
            return data1, np.zeros_like(incoming_grads)
        return [data1]
    def check_bilinear_resize_op(shape, height, width):
        x = mx.nd.random.uniform(shape=shape)
        y = mx.nd.contrib.BilinearResize2D(x, height=height, width=width)
        assert_almost_equal(y, py_bilinear_resize(x.asnumpy(), height, width))

        x_scale = width / shape[-1]
        y_scale = height / shape[-2]
        y = mx.nd.contrib.BilinearResize2D(x, scale_height=y_scale, scale_width=x_scale)
        assert_almost_equal(y, py_bilinear_resize(x.asnumpy(), height, width))
    def check_bilinear_resize_modes_op(shape, scale_height=None, scale_width=None, shape_1=None, mode=None):
        x = mx.nd.random.uniform(shape=shape)
        original_h = shape[2]
        original_w = shape[3]
        if mode == 'odd_scale':
            assert scale_height is not None and scale_width is not None
            new_h = int(original_h * scale_height) if (original_h % 2) == 0 else \
                int((original_h - 1) * scale_height) + 1
            new_w = int(original_w * scale_width) if (original_w % 2) == 0 \
                else int((original_w - 1) * scale_width) + 1
            y = mx.nd.contrib.BilinearResize2D(x, scale_height=scale_height,
                                               scale_width=scale_width,
                                               mode='odd_scale')
        elif mode == 'to_even_down':
            new_h = original_h if (original_h % 2) == 0 else original_h - 1
            new_w = original_w if (original_w % 2) == 0 else original_w - 1
            y = mx.nd.contrib.BilinearResize2D(x, mode='to_even_down')
        elif mode == 'to_even_up':
            new_h = original_h if (original_h % 2) == 0 else original_h + 1
            new_w = original_w if (original_w % 2) == 0 else original_w + 1
            y = mx.nd.contrib.BilinearResize2D(x, mode='to_even_up')
        elif mode == 'to_odd_down':
            new_h = original_h if (original_h % 2) == 1 else original_h - 1
            new_w = original_w if (original_w % 2) == 1 else original_w - 1
            y = mx.nd.contrib.BilinearResize2D(x, mode='to_odd_down')
        elif mode == 'to_odd_up':
            new_h = original_h if (original_h % 2) == 1 else original_h + 1
            new_w = original_w if (original_w % 2) == 1 else original_w + 1
            y = mx.nd.contrib.BilinearResize2D(x, mode='to_odd_up')
        elif mode == 'like':
            x_1 = mx.nd.random.uniform(shape=shape_1)
            new_h = x_1.shape[2]
            new_w = x_1.shape[3]
            y = mx.nd.contrib.BilinearResize2D(x, x_1, mode='like')
        new_shape_desired = np.array([shape[0], shape[1], new_h, new_w], dtype='int')
        new_shape_got = np.array(y.shape, dtype='int')
        data_sym = mx.sym.var('data')
        data_np = x.asnumpy()
        expected = py_bilinear_resize(data_np, new_h, new_w)
        out_grads = np.ones([shape[0], shape[1], new_h, new_w])
        expected_backward = py_bilinear_resize_backward(data_np, out_grads, mode)
        assert_array_equal(new_shape_desired, new_shape_got, "Desired and got shapes are not equal. {} vs {}".format(
            str(new_shape_desired.tolist()), str(new_shape_got.tolist())))
        assert_almost_equal(y.asnumpy(), expected, 1e-3, 0)
        if mode != 'like':
            resize_sym = mx.sym.contrib.BilinearResize2D(data_sym, None, scale_height=scale_height, scale_width=scale_width, mode=mode)
            check_symbolic_forward(resize_sym, [data_np], [expected], rtol=1e-3, atol=1e-5)
            check_symbolic_backward(resize_sym, [data_np], [out_grads], expected_backward, rtol=1e-3, atol=1e-5)
            check_numeric_gradient(resize_sym, [data_np], rtol=1e-2, atol=1e-4)
        else:
            data_sym_like = mx.sym.var('data_like')
            resize_sym = mx.sym.contrib.BilinearResize2D(data_sym, data_sym_like, mode=mode)
            date_np_like = x_1.asnumpy()
            check_symbolic_forward(resize_sym, [data_np, date_np_like], [expected], rtol=1e-3, atol=1e-5)
            check_symbolic_backward(resize_sym, [data_np, date_np_like], [out_grads], expected_backward, rtol=1e-3, atol=1e-5)
            check_numeric_gradient(resize_sym, [data_np, date_np_like], rtol=1e-2, atol=1e-4)

    shape = (2, 2, 10, 10)
    check_bilinear_resize_op(shape, 5, 5)
    print('correct')
    check_bilinear_resize_op(shape, 10, 10)
    print('correct')
    check_bilinear_resize_op(shape, 15, 15)
    print('correct')
    check_bilinear_resize_op(shape, 3, 7)
    print('correct')
    check_bilinear_resize_op(shape, 13, 17)
    print('correct')
    shape = (2, 2, 20, 20)
    check_bilinear_resize_modes_op(shape, scale_height=0.5, scale_width=0.5, mode='odd_scale')
    print('correct')
    check_bilinear_resize_modes_op(shape, scale_height=5, scale_width=10, mode='odd_scale')
    print('correct')
    check_bilinear_resize_modes_op(shape, scale_height=0.1, scale_width=0.2, mode='odd_scale')
    print('correct')
    check_bilinear_resize_modes_op(shape, mode='to_even_down')
    print('correct')
    check_bilinear_resize_modes_op(shape, mode='to_even_up')
    print('correct')
    check_bilinear_resize_modes_op(shape, mode='to_odd_down')
    print('correct')
    check_bilinear_resize_modes_op(shape, mode='to_odd_up')
    print('correct')
    shape = (2, 2, 21, 21)
    check_bilinear_resize_modes_op(shape, scale_height=0.5, scale_width=0.5, mode='odd_scale')
    print('correct')
    check_bilinear_resize_modes_op(shape, scale_height=5, scale_width=10, mode='odd_scale')
    print('correct')
    check_bilinear_resize_modes_op(shape, scale_height=0.1, scale_width=0.2, mode='odd_scale')
    print('correct')
    check_bilinear_resize_modes_op(shape, mode='to_even_down')
    print('correct')
    check_bilinear_resize_modes_op(shape, mode='to_even_up')
    print('correct')
    check_bilinear_resize_modes_op(shape, mode='to_odd_down')
    print('correct')
    check_bilinear_resize_modes_op(shape, mode='to_odd_up')
    print('correct')
    shape_0 = (2, 2, 21, 21)
    shape_1 = (2, 2, 10, 10)
    check_bilinear_resize_modes_op(shape_0, shape_1=shape_1, mode='like')
    print('correct')
    check_bilinear_resize_modes_op(shape_1, shape_1=shape_0, mode='like')
    print('correct')
    print('all correct')


def test_image_resize():
    # image = mx.nd.random.uniform(0, 255, (4, 2, 3)).astype(dtype=np.uint8)
    image = mx.nd.array(
        [[[139, 151, 182],
          [215, 153, 218]],
         [[138, 216, 108],
          [159, 164,  98]],
         [[111,  75, 227],
          [14, 245,  69]],
         [[97, 121, 201],
          [207, 134, 122]]]).astype(dtype=np.uint8)
    print(image)
    print('the result:')
    print(mx.nd.image.resize(image, (3, 3)))
    print('''
        excepted:
        [[[139 151 182]
        [177 152 200]
        [215 153 218]]

        [[124 145 167]
        [105 175 125]
        [ 86 204  83]]

        [[ 97 121 201]
        [152 127 161]
        [207 134 122]]]
<NDArray 3x3x3 @cpu(0)>
        ''')
    # image = mx.nd.random.uniform(0, 255, (2, 4, 2, 3)).astype(dtype=np.uint8)
    image = mx.nd.array(
        [[[[139, 151, 182],
           [215, 153, 218]],
          [[138, 216, 108],
           [159, 164,  98]],
          [[111,  75, 227],
           [14, 245,  69]],
          [[97, 121, 201],
           [207, 134, 122]]], 
         [[[144, 100, 236],
           [213,  18,  86]],
          [[22, 165,   5],
           [93, 212, 244]],
          [[198,  35, 221],
           [221, 249, 120]],
          [[203, 204, 117],
           [132, 199, 173]]]]).astype(dtype=np.uint8)
    print(image)
    print('the result:')
    print(mx.nd.image.resize(image, (2, 2)))
    print('''
        excepted:
        [[[[139 151 182]
        [215 153 218]]

        [[ 97 121 201]
        [207 134 122]]]


        [[[144 100 236]
        [213  18  86]]

        [[203 204 117]
        [132 199 173]]]]
        <NDArray 2x2x2x3 @cpu(0)>
        ''')
def my_test():        
    # test_image_resize()
    test_bilinear_resize_op()
    


if __name__ == '__main__':
    my_test()
    
        
    
    
