import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.base import py_str, MXNetError, _as_list
from mxnet.test_utils import check_consistency, set_default_context, assert_almost_equal, assert_allclose

def default_context():
    return mx.cpu()

def with_seed(*args, **kwargs):
  def decorator(func):
    return func
  return decorator

def assert_exception(f, exception_type, *args, **kwargs):
    """Test that function f will throw an exception of type given by `exception_type`"""
    try:
        f(*args, **kwargs)
        assert(False)
    except exception_type:
        return

def check_pad_with_shape(shape, xpu, pad_width, mode, dtype="float64"):
    # bind with label
    X = mx.symbol.Variable('X', dtype=dtype)
    Y = mx.symbol.Pad(data=X, mode=mode, pad_width=pad_width)
    x = mx.random.uniform(-1, 1, shape, ctx=mx.cpu(), dtype=dtype).copyto(xpu)
    # numpy result
    pad_grouped = list(zip(*[iter(list(pad_width))] * 2))
    np_out = np.pad(x.asnumpy(), pad_grouped, mode)
    # mxnet result
    grad = mx.nd.empty(shape, ctx = xpu, dtype=dtype)
    exec1 = Y.bind(xpu, args = [x], args_grad = {'X': grad})
    exec1.forward(is_train=True)
    out = exec1.outputs[0]
    # compare numpy + mxnet
    assert_almost_equal(out, np_out)
    # grad check
    check_numeric_gradient(Y, [x.asnumpy()], numeric_eps=1e-2, rtol=1e-2)


@with_seed()
def test_pad():
    ctx = default_context()
    shape1 = (2, 3, 3, 5)
    pad1 = (0, 0, 0, 0, 1, 2, 3, 4)
    shape2 = (2, 3, 3, 5, 4)
    pad2 = (0, 0, 0, 0, 1, 2, 3, 4, 3, 1)
    # note: this op doesn't support ints yet. Add tests when supported
    dtypes = ["float16", "float32", "float64"]
    for dtype in dtypes:
        check_pad_with_shape(shape1, ctx, pad1, 'constant', dtype)
        check_pad_with_shape(shape1, ctx, pad1, 'edge', dtype)
        check_pad_with_shape(shape2, ctx, pad2, 'constant', dtype)
        check_pad_with_shape(shape2, ctx, pad2, 'edge', dtype)
        check_pad_with_shape(shape1, ctx, pad1, 'reflect', dtype)
        check_pad_with_shape(shape2, ctx, pad2, 'reflect', dtype)


test_pad()