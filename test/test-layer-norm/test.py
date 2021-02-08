import os
import mxnet as mx
import numpy as np
from mxnet import test_utils
from mxnet.test_utils import assert_almost_equal check_numeric_gradient

def default_context():
    return mx.cpu()


def check_layer_normalization(in_shape, axis, eps, dtype=np.float32,
                              forward_check_eps=1E-3, backward_check_eps=1E-3,
                              npy_grad_check=True, finite_grad_check=True):
    def npy_layer_norm(data, gamma, beta, axis=1, eps=1E-5):
        if axis < 0:
            axis += data.ndim
        broadcast_shape = [1 for _ in range(data.ndim)]
        broadcast_shape[axis] = data.shape[axis]
        mean = data.mean(axis=axis, keepdims=True).astype(dtype)
        var = data.var(axis=axis, keepdims=True).astype(dtype)
        std = np.sqrt(var + dtype(eps)).astype(dtype)
        out = np.reshape(gamma, broadcast_shape) * (data - mean) / std + \
              np.reshape(beta, broadcast_shape)
        return out

    def npy_layer_norm_grad(data, gamma, out_grad, axis, eps):
        if axis < 0:
            axis += data.ndim
        exclude_axis = tuple([ele for ele in range(data.ndim) if ele != axis])
        data_mean = data.mean(axis=axis, keepdims=True)
        data_var = data.var(axis=axis, keepdims=True)
        data_std = np.sqrt(data_var + eps)
        centered_data = (data - data_mean) / data_std
        gamma_grad = (centered_data * out_grad).sum(axis=exclude_axis, keepdims=True)
        beta_grad = out_grad.sum(axis=exclude_axis, keepdims=True)
        w = out_grad * gamma.reshape([1 if i != axis else data.shape[axis] for i in range(data.ndim)])\
            / data_std
        data_grad = w - w.mean(axis=axis, keepdims=True)\
                    - centered_data * (w * centered_data).mean(axis=axis, keepdims=True)
        gamma_grad = gamma_grad.reshape((-1,))
        beta_grad = beta_grad.reshape((-1,))
        return data_grad, gamma_grad, beta_grad

    ctx = default_context()
    data = np.random.normal(0, 1, in_shape).astype(dtype)
    gamma = np.random.normal(0, 1, (in_shape[axis],)).astype(dtype)
    beta = np.random.normal(0, 1, (in_shape[axis],)).astype(dtype)
    data_s = mx.symbol.Variable('data')
    gamma_s = mx.symbol.Variable('gamma')
    beta_s = mx.symbol.Variable('beta')
    out_s = mx.symbol.LayerNorm(data=data_s, gamma=gamma_s, beta=beta_s, axis=axis, eps=eps)
    exe = out_s.simple_bind(ctx, data=in_shape)
    exe.arg_dict['data'][:] = data
    exe.arg_dict['gamma'][:] = gamma
    exe.arg_dict['beta'][:] = beta
    out_nd = exe.forward()[0]
    out = npy_layer_norm(data, gamma, beta, axis, eps)
    print('check out')
    assert_almost_equal(out, out_nd, forward_check_eps, forward_check_eps)

    if finite_grad_check:
        for req in ['write', 'add']:
            check_numeric_gradient(out_s, {'data': data, 'gamma': gamma, 'beta': beta},
                                   grad_nodes={'data': req, 'gamma': req, 'beta': req},
                                   numeric_eps=1e-2, rtol=1e-2, atol=1e-2)

    if npy_grad_check:
        # Test for grad_req = write
        out_grad = np.random.normal(0, 1, in_shape).astype(dtype)
        exe = out_s.simple_bind(ctx, data=in_shape, grad_req='write')
        exe.arg_dict['data'][:] = data
        exe.arg_dict['gamma'][:] = gamma
        exe.arg_dict['beta'][:] = beta
        exe.forward()
        exe.backward([mx.nd.array(out_grad, ctx=ctx)])
        gt_data_grad, gt_gamma_grad, gt_beta_grad =\
            npy_layer_norm_grad(data, gamma, out_grad, axis, eps)
        print('check data')
        assert_almost_equal(exe.grad_dict['data'].asnumpy(), gt_data_grad, backward_check_eps, backward_check_eps)
        print('check gamma')
        assert_almost_equal(exe.grad_dict['gamma'].asnumpy(), gt_gamma_grad, backward_check_eps, backward_check_eps)
        print('check beta')
        assert_almost_equal(exe.grad_dict['beta'].asnumpy(), gt_beta_grad, backward_check_eps, backward_check_eps)

        # Test for grad_req = add
        out_grad = np.random.normal(0, 1, in_shape).astype(dtype)
        init_data_grad = np.random.normal(0, 1, in_shape).astype(dtype)
        init_gamma_grad = np.random.normal(0, 1, (in_shape[axis],)).astype(dtype)
        init_beta_grad = np.random.normal(0, 1, (in_shape[axis],)).astype(dtype)
        exe = out_s.simple_bind(ctx, data=in_shape, grad_req='add')
        exe.arg_dict['data'][:] = data
        exe.arg_dict['gamma'][:] = gamma
        exe.arg_dict['beta'][:] = beta
        exe.grad_dict['data'][:] = init_data_grad
        exe.grad_dict['gamma'][:] = init_gamma_grad
        exe.grad_dict['beta'][:] = init_beta_grad
        exe.forward()
        exe.backward([mx.nd.array(out_grad, ctx=ctx)])
        gt_data_grad, gt_gamma_grad, gt_beta_grad = \
            npy_layer_norm_grad(data, gamma, out_grad, axis, eps)
        assert_almost_equal(exe.grad_dict['data'].asnumpy(),
                            gt_data_grad + init_data_grad, backward_check_eps, backward_check_eps)
        assert_almost_equal(exe.grad_dict['gamma'].asnumpy(),
                            gt_gamma_grad + init_gamma_grad, backward_check_eps, backward_check_eps)
        assert_almost_equal(exe.grad_dict['beta'].asnumpy(),
                            gt_beta_grad + init_beta_grad, backward_check_eps, backward_check_eps)


def test_layer_norm():
    for enforce_safe_acc in ["1", "0"]:
        os.environ["MXNET_SAFE_ACCUMULATION"] = enforce_safe_acc
        for dtype, forward_check_eps, backward_check_eps in zip([np.float16, np.float32, np.float64],
                                                                [1E-2, 1E-3, 1E-4],
                                                                [1E-2, 1E-3, 1E-4]):
            if dtype != np.float16:
                in_shape_l, finite_grad_check_l = [(10, 6, 5), (10, 10), (128 * 32, 512)], [True, True, False]
            else:
                in_shape_l, finite_grad_check_l = [(10, 6, 5), (10, 10)], [True, True]  # large input + fp16 does not pass the forward check
            for in_shape, finite_grad_check in zip(in_shape_l, finite_grad_check_l):
                for axis in range(-len(in_shape), len(in_shape)):
                    for eps in [1E-2, 1E-3]:
                        if dtype == np.float16:
                            npy_grad_check = False
                        else:
                            npy_grad_check = True
                        print(f'dtype={dtype}, forward_check_eps={forward_check_eps}, backward_check_eps={backward_check_eps}, '
                            f'in_shape={in_shape}, finite_grad_check={finite_grad_check}, '
                            f'axis={axis}, eps={eps}, npy_grad_check={npy_grad_check}')
                        check_layer_normalization(in_shape, axis, eps, dtype=dtype,
                                                  forward_check_eps=forward_check_eps,
                                                  backward_check_eps=backward_check_eps,
                                                  npy_grad_check=npy_grad_check,
                                                  finite_grad_check=finite_grad_check)


if __name__ == '__main__':
    test_layer_norm()
    
        
    
    