import numbers
import sys
import os
import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.base import py_str, MXNetError, _as_list
from mxnet.test_utils import set_default_context, assert_almost_equal, assert_allclose, check_numeric_gradient
from pickle import load, dump


output_file = sys.argv[1] if len(sys.argv) > 1 else "output"
input_file = sys.argv[2] if len(sys.argv) > 2 else "input"


def with_seed(*args, **kwargs):
  def decorator(func):
    return func
  return decorator


@with_seed()
def test_adaptive_avg_pool_op():
    def py_adaptive_avg_pool(x, height, width):
        # 2D per frame adaptive avg pool
        def adaptive_avg_pool_frame(x, y):
            isizeH, isizeW = x.shape
            osizeH, osizeW = y.shape
            for oh in range(osizeH):
                istartH = int(np.floor(1.0 * (oh * isizeH) / osizeH))
                iendH = int(np.ceil(1.0 * (oh + 1) * isizeH / osizeH))
                kH = iendH - istartH
                for ow in range(osizeW):
                    istartW = int(np.floor(1.0 * (ow * isizeW) / osizeW))
                    iendW = int(np.ceil(1.0 * (ow + 1) * isizeW / osizeW))
                    kW = iendW - istartW
                    xsum = 0
                    for ih in range(kH):
                        for iw in range(kW):
                            xsum += x[istartH+ih][istartW+iw]
                    y[oh][ow] = xsum / kH / kW

        B,C,_,_ = x.shape
        y = np.empty([B,C,height, width], dtype=x.dtype)
        for b in range(B):
            for c in range(C):
                adaptive_avg_pool_frame(x[b][c], y[b][c])
        return y
    def check_adaptive_avg_pool_op(shape, output_height, output_width=None):
        if fin.readable():
            # 从前面的记录里读取
            x = load(fin)
        else:
            x = mx.nd.random.uniform(shape=shape)
            dump(x, fin)
        if output_width is None:
            x.attach_grad()
            with mx.autograd.record():
                y = mx.nd.contrib.AdaptiveAvgPooling2D(x, output_size=output_height)
            y.backward()
            if fout.readable():
                assert_almost_equal(x.grad, load(fout))
            else:
                dump(x.grad, fout)
            npy = py_adaptive_avg_pool(x.asnumpy(), output_height, output_height)
        else:
            x.attach_grad()
            with mx.autograd.record():
                y = mx.nd.contrib.AdaptiveAvgPooling2D(x, output_size=(output_height, output_width))
            y.backward()
            if fout.readable():
                assert_almost_equal(x.grad, load(fout))
            else:
                dump(x.grad, fout)
            npy = py_adaptive_avg_pool(x.asnumpy(), output_height, output_width)
        assert_almost_equal(y.asnumpy(), npy)
    shape = (2, 2, 10, 10)
    for i in range(1, 11):
        check_adaptive_avg_pool_op(shape, i)
        for j in range(1, 11):
            check_adaptive_avg_pool_op(shape, i, j)

def print_args(func):
    def decorator(*args, **kwargs):
        print(*args,sep='\n')
        for k, v in kwargs:
            print(k, '=', v)
        func(*args, **kwargs)
    return decorator



def my_test():
    test_adaptive_avg_pool_op()

if __name__ == '__main__':
    in_mode = "rb" if os.path.exists(input_file) else "wb"
    out_mode = "rb" if "r" in in_mode and os.path.exists(output_file) else "wb"
    print("{} input in file '{}'".format("use old" if "r" in in_mode else "store random", input_file))
    print("{} output in file '{}'".format("compare" if "r" in out_mode else "store", output_file))
    with open(input_file, in_mode) as fin, open(output_file, out_mode) as fout:
        my_test()
    try:
        print(f"print {input_file} in {input_file + '.txt'}")
        with open(input_file, "rb") as fin, open(input_file + ".txt", "w") as in_print:
            while True:
                item = load(fin)
                in_print.write(str(item))
                in_print.write('\n')
    except EOFError:
        pass
    try:
        output_print_file = output_file + ("2" if "r" in out_mode else "1") + ".txt"
        print(f"print {output_file} in {output_print_file}")
        with open(output_file, "rb") as fout, open(output_print_file, "w") as out_print:
            out_print.write(f"in {output_print_file}\n")
            while True:
                item = load(fout)
                out_print.write(str(item))
                out_print.write('\n')
    except EOFError:
        pass
    
        
    
    