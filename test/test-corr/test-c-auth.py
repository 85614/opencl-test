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

@with_seed()
def test_invalid_kernel_size():
    invalid_kernel_size = 28
    assert_exception(
        mx.nd.Correlation,
        MXNetError,
        mx.nd.array(np.random.rand(1, 1, 28, 28)),
        mx.nd.array(np.random.rand(1, 1, 28, 28)),
        kernel_size=invalid_kernel_size)

@with_seed()
def test_valid_kernel_size():
    valid_kernel_size = 9
    mx.nd.Correlation(
        mx.nd.array(np.random.rand(1, 1, 28, 28)),
        mx.nd.array(np.random.rand(1, 1, 28, 28)),
        kernel_size=valid_kernel_size)


def get_correlation(data1,data2,kernel_size,max_displacement,stride1,stride2,pad_size,is_multiply):

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')
    return mx.sym.Correlation(data1=img1,data2=img2,kernel_size =kernel_size,max_displacement = max_displacement,
                              stride1 = stride1,stride2 = stride2,pad_size= pad_size,is_multiply = is_multiply)


def correlation_forward(data1,data2,pad_size,kernel_size,stride1,stride2,max_displacement,is_multiply):

    # compute output's dimension
    paddedbottomheight = data1.shape[2] + 2 * pad_size
    paddedbottomwidth = data1.shape[3] + 2 * pad_size
    kernel_radius = (kernel_size - 1) // 2
    border_size = max_displacement + kernel_radius
    top_width = (paddedbottomwidth - border_size * 2) // stride1
    top_height = (paddedbottomheight - border_size  * 2) // stride1
    neighborhood_grid_radius = max_displacement // stride2
    neighborhood_grid_width = neighborhood_grid_radius * 2 + 1
    top_channels = neighborhood_grid_width * neighborhood_grid_width

    out = np.zeros((data1.shape[0], top_channels, top_height, top_width))
    tmp1 = np.zeros((data1.shape[0],data1.shape[1],paddedbottomheight, paddedbottomwidth))
    tmp2 = np.zeros((data1.shape[0],data1.shape[1],paddedbottomheight, paddedbottomwidth))

    tmp1[:, :, pad_size:pad_size + data1.shape[2], pad_size:pad_size + data1.shape[3]] = data1[:,:,:,:]
    tmp2[:, :, pad_size:pad_size + data2.shape[2], pad_size:pad_size + data2.shape[3]] = data2[:,:,:,:]

    for i in range(top_height):
        for j in range(top_width):
            for nbatch in range(data1.shape[0]):

                # x1,y1 is the location in data1 , i,j is the location in output
                x1 = j * stride1 + max_displacement
                y1 = i * stride1 + max_displacement

                for top_channel in range(top_channels):

                    s2o = (top_channel % neighborhood_grid_width - neighborhood_grid_radius) * stride2
                    s2p = (top_channel // neighborhood_grid_width - neighborhood_grid_radius) * stride2

                    # location in data2
                    x2 = x1 + s2o
                    y2 = y1 + s2p

                    for h in range(kernel_size):
                        for w in range(kernel_size):
                            for channel in range(data1.shape[1]):
                                if is_multiply:
                                    out[nbatch, top_channel, i, j] += tmp1[nbatch, channel,y1 + h, x1 + w] * tmp2[nbatch, channel, y2 + h,x2 + w]
                                else:
                                    out[nbatch, top_channel, i, j] += abs(tmp1[nbatch, channel, y1 + h, x1 + w] - tmp2[nbatch, channel, y2 + h, x2 + w])
    out /= float(kernel_size**2*data1.shape[1])
    return out,tmp1,tmp2


def correlation_backward(out_grad,tmp1,tmp2,data1,data2,pad_size,kernel_size,stride1,stride2,max_displacement,is_multiply):

    # compute output's dimension
    paddedbottomheight = data1.shape[2] + 2 * pad_size
    paddedbottomwidth = data1.shape[3] + 2 * pad_size
    kernel_radius = (kernel_size - 1) // 2
    border_size = max_displacement + kernel_radius
    top_width = (paddedbottomwidth - border_size * 2) // stride1
    top_height = (paddedbottomheight - border_size  * 2) // stride1
    neighborhood_grid_radius = max_displacement // stride2
    neighborhood_grid_width = neighborhood_grid_radius * 2 + 1
    top_channels = neighborhood_grid_width * neighborhood_grid_width

    out = np.zeros((data1.shape[0], top_channels, top_height, top_width))
    tmp1_grad = np.zeros(tmp1.shape)
    tmp2_grad = np.zeros(tmp2.shape)

    for i in range(top_height):
        for j in range(top_width):
            for nbatch in range(data1.shape[0]):

                # x1,y1 is the location in data1 , i,j is the location in output
                x1 = j * stride1 + max_displacement
                y1 = i * stride1 + max_displacement

                for top_channel in range(top_channels):

                    s2o = (top_channel % neighborhood_grid_width - neighborhood_grid_radius) * stride2
                    s2p = (top_channel // neighborhood_grid_width - neighborhood_grid_radius) * stride2

                    # location in data2
                    x2 = x1 + s2o
                    y2 = y1 + s2p

                    for h in range(kernel_size):
                        for w in range(kernel_size):
                            for channel in range(data1.shape[1]):
                                if is_multiply:
                                    tmp1_grad[nbatch,channel,y1+h,x1+w]+= out_grad[nbatch,top_channel,i,j]*tmp2[nbatch, channel, y2 + h,x2 + w]
                                    tmp2_grad[nbatch,channel,y2+h,x2+w]+= out_grad[nbatch,top_channel,i,j]*tmp1[nbatch, channel, y1 + h,x1 + w]
                                else:
                                    sgn = 1 if (tmp1[nbatch, channel, y1 + h,x1 + w]>=tmp2[nbatch, channel, y2 + h,x2 + w]) else -1
                                    tmp1_grad[nbatch,channel,y1+h,x1+w]+= out_grad[nbatch,top_channel,i,j]*sgn
                                    tmp2_grad[nbatch,channel,y2+h,x2+w]+= out_grad[nbatch,top_channel,i,j]*(-sgn)

    tmp1_grad = tmp1_grad / float(kernel_size**2*data1.shape[1])
    tmp2_grad = tmp2_grad / float(kernel_size**2*data1.shape[1])
    return tmp1_grad[:,:,pad_size:pad_size+data1.shape[2],pad_size:pad_size+data1.shape[3]],tmp2_grad[:,:,pad_size:pad_size+data1.shape[2],pad_size:pad_size+data1.shape[3]],


def unittest_correlation(data_shape,kernel_size,max_displacement,stride1,stride2,pad_size,is_multiply,dtype):

    img1 = np.random.random(data_shape)
    img1 = img1.astype(dtype)
    img2 = np.random.random(data_shape)
    img2 = img2.astype(dtype)

    net1 = get_correlation(img1,img2,kernel_size,max_displacement,stride1,stride2,pad_size,is_multiply)
    net2 = get_correlation(img1,img2,kernel_size,max_displacement,stride1,stride2,pad_size,is_multiply )

    exe1 = net1.simple_bind(default_context(),img1=img1.shape,img2=img1.shape)
    exe1.arg_dict['img1'][:] = img1
    exe1.arg_dict['img2'][:] = img2

    #cpu forward
    exe1.forward(is_train=True)
    # python forward
    forward_result,tmp1,tmp2 = correlation_forward(img1,img2,pad_size,kernel_size,stride1,stride2,max_displacement,is_multiply)

    # forward error
    assert_almost_equal(exe1.outputs[0], forward_result, rtol=1e-4, atol=1e-4)

    # out_grad
    a = np.ones(forward_result.shape)
    out_grad1 = mx.nd.array(a,default_context())
    # cpu backward
    exe1.backward(out_grads=out_grad1)
    # python backward
    grad1,grad2 = correlation_backward(a,tmp1,tmp2,img1,img2,pad_size,kernel_size,stride1,stride2,max_displacement,is_multiply)

    # backward error
    assert_almost_equal(exe1.grad_dict['img1'], grad1, rtol=1e-3, atol=1e-4)
    assert_almost_equal(exe1.grad_dict['img2'], grad2, rtol=1e-3, atol=1e-4)


@with_seed()
def test_correlation():
    def test_infer_type(dtype):
        a = mx.sym.Variable('a')
        b = mx.sym.Variable('b')
        corr = mx.sym.Correlation(data1=a, data2=b)
        arg_type1, out_type1, _ = corr.infer_type(a=dtype)
        if arg_type1[0] != np.dtype(dtype) and arg_type1[1] != np.dtype(dtype) and out_type1[0] != np.dtype(dtype):
            msg = npt.npt.build_err_msg([a, b],
                                        err_msg="Inferred type from a is not as expected, "
                                                "Expected :%s %s %s, Got: %s %s %s"
                                                % (dtype, dtype, dtype, arg_type1[0], arg_type1[1], out_type1[0]),
                                                names=['a', 'b'])
            raise AssertionError(msg)
        arg_type2, out_type2, _ = corr.infer_type(b=dtype)
        if arg_type2[0] != np.dtype(dtype) and arg_type2[1] != np.dtype(dtype) and out_type2[0] != np.dtype(dtype):
            msg = npt.npt.build_err_msg([a, b],
                                        err_msg="Inferred type from b is not as expected, "
                                                "Expected :%s %s %s, Got: %s %s %s"
                                                % (dtype, dtype, dtype, arg_type1[0], arg_type1[1], out_type1[0]),
                                                names=['a', 'b'])
            raise AssertionError(msg)

    for dtype in ['float16', 'float32']:
        test_infer_type(dtype)
        unittest_correlation((1,3,10,10), kernel_size = 1,max_displacement = 4,stride1 = 1,stride2 = 1,pad_size = 4,is_multiply = False, dtype = dtype)        
        unittest_correlation((5,1,15,15), kernel_size = 1,max_displacement = 5,stride1 = 1,stride2 = 1,pad_size = 5,is_multiply = False, dtype = dtype)
        unittest_correlation((5,1,15,15), kernel_size = 1,max_displacement = 5,stride1 = 1,stride2 = 1,pad_size = 5,is_multiply = True, dtype = dtype)
        unittest_correlation((5,1,15,15), kernel_size = 1,max_displacement = 10,stride1 = 1,stride2 = 2,pad_size = 10,is_multiply = True, dtype = dtype)
        unittest_correlation((5,1,4,4), kernel_size = 3,max_displacement = 1,stride1 = 1,stride2 = 1,pad_size = 2,is_multiply = True, dtype = dtype)
        unittest_correlation((5,1,4,4), kernel_size = 3,max_displacement = 1,stride1 = 2,stride2 = 1,pad_size = 2,is_multiply = True, dtype = dtype)
        unittest_correlation((5,1,4,4), kernel_size = 3,max_displacement = 1,stride1 = 2,stride2 = 1,pad_size = 2,is_multiply = False, dtype = dtype)
        unittest_correlation((5,1,6,4), kernel_size = 3,max_displacement = 1,stride1 = 2,stride2 = 1,pad_size = 2,is_multiply = False, dtype = dtype)
        unittest_correlation((5,1,11,11), kernel_size = 5,max_displacement = 1,stride1 = 1,stride2 = 1,pad_size = 2,is_multiply = False, dtype = dtype)


test_correlation()
test_valid_kernel_size()
test_invalid_kernel_size()

