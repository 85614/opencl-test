import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.test_utils import check_consistency, set_default_context, assert_almost_equal, assert_allclose

def default_context():
    return mx.cpu()

def test_bilinear_sampler_with_type():
    data = mx.sym.Variable('data')
    grid = mx.sym.Variable('grid')
    sym = mx.sym.BilinearSampler(data=data, grid=grid)
    ctx_list = [{'ctx': mx.cpu(0), 'data': (1, 5, 10, 10), 'grid': (1, 2, 10, 10),
                  'type_dict': {'data': np.float64}},
                {'ctx': mx.cpu(0), 'data': (1, 5, 10, 10), 'grid': (1, 2, 10, 10),
                  'type_dict': {'data': np.float32}}]
    check_consistency(sym, ctx_list)
    check_consistency(sym, ctx_list, grad_req="add")



def test_bilinear_sampler_versions():
    data = mx.sym.Variable('data')
    grid = mx.sym.Variable('grid')
    sym1 = mx.sym.BilinearSampler(data=data, grid=grid)
    sym2 = mx.sym.BilinearSampler(data=data, grid=grid, cudnn_off=True)
    sym3 = mx.sym.BilinearSampler(data=data, grid=grid)

    test_cases = [[(1,3,15,16),(1,2,10,10)],
                 [(1,6,7,16),(1,2,10,4)],
                 [(1,7,3,16),(1,2,8,11)],
                 [(1,9,50,50),(1,2,50,50)]]

    for item in test_cases:
        data_shape, grid_shape = item
        # kWriteTo
        exe_cpu = sym1.simple_bind(data=data_shape, grid=grid_shape, ctx=mx.cpu(), grad_req='write')
        exe_gpu = sym2.simple_bind(data=data_shape, grid=grid_shape, ctx=default_context(), grad_req='write')
        exe_cudnn = sym3.simple_bind(data=data_shape, grid=grid_shape, ctx=default_context(), grad_req='write')
        exe_list = [exe_cpu, exe_gpu, exe_cudnn]
        ref_idx = 0
        test_data = np.random.uniform(low=-0.1, high=0.1,size=data_shape).astype(np.float32)
        test_grid = np.random.uniform(low=-2, high=2, size=grid_shape).astype(np.float32)
        for exe in exe_list:
            exe.arg_dict['data'][:] = test_data
            exe.arg_dict['grid'][:] = test_grid
            exe.forward(is_train=True)
            mx.test_utils.assert_almost_equal(exe_list[ref_idx].outputs[0], exe.outputs[0], rtol=1e-3, atol=1e-5)

        out_grad = np.random.uniform(low=-0.01, high=0.01,size=data_shape[:2] + grid_shape[2:]).astype(np.float32)
        for exe in exe_list:
            exe.backward(mx.nd.array(out_grad))
            assert_almost_equal(exe.grad_dict['data'], exe_list[ref_idx].grad_dict['data'], rtol=1e-3, atol=1e-5)
            assert_almost_equal(exe.grad_dict['grid'], exe_list[ref_idx].grad_dict['grid'], rtol=1e-3, atol=1e-5)

        data_grad = exe_list[ref_idx].grad_dict['data'].asnumpy()
        grid_grad = exe_list[ref_idx].grad_dict['grid'].asnumpy()

        # kAddTo
        exe_cpu_addto = sym1.simple_bind(data=data_shape, grid=grid_shape, ctx=mx.cpu(), grad_req='add')
        exe_gpu_addto = sym2.simple_bind(data=data_shape, grid=grid_shape, ctx=default_context(), grad_req='add')
        exe_cudnn_addto = sym3.simple_bind(data=data_shape, grid=grid_shape, ctx=default_context(), grad_req='add')
        exe_list = [exe_cpu_addto, exe_gpu_addto, exe_cudnn_addto]
        data_initial_grad = np.random.normal(size=exe_list[ref_idx].grad_dict['data'].shape).astype(np.float32)
        grid_initial_grad = np.random.normal(size=exe_list[ref_idx].grad_dict['grid'].shape).astype(np.float32)
        for exe in exe_list:
            exe.arg_dict['data'][:] = test_data
            exe.arg_dict['grid'][:] = test_grid
            exe.grad_dict['data'][:] = data_initial_grad
            exe.grad_dict['grid'][:] = grid_initial_grad
            exe.forward(is_train=True)
            exe.backward(mx.nd.array(out_grad))
            assert_almost_equal(exe.grad_dict['data'], exe_list[ref_idx].grad_dict['data'], rtol=1e-3, atol=1e-5)
            assert_almost_equal(exe.grad_dict['grid'], exe_list[ref_idx].grad_dict['grid'], rtol=1e-3, atol=1e-5)
        assert_almost_equal(exe_list[ref_idx].grad_dict['data'], data_grad + data_initial_grad, rtol=1e-3, atol=1e-5)
        assert_almost_equal(exe_list[ref_idx].grad_dict['grid'], grid_grad + grid_initial_grad, rtol=1e-3, atol=1e-5)

        for req_dict in [{'data' : 'null', 'grid' : 'write'}, {'data' : 'write', 'grid' : 'null'}]:
            # Mixture of kWriteTo and kNullOp
            exe_cpu_mix = sym1.simple_bind(data=data_shape, grid=grid_shape, ctx=mx.cpu(), grad_req=req_dict)
            exe_gpu_mix = sym2.simple_bind(data=data_shape, grid=grid_shape, ctx=default_context(), grad_req=req_dict)
            exe_cudnn_mix = sym3.simple_bind(data=data_shape, grid=grid_shape, ctx=default_context(), grad_req=req_dict)
            exe_list = [exe_cpu_mix, exe_gpu_mix, exe_cudnn_mix]
            for exe in exe_list:
                exe.arg_dict['data'][:] = test_data
                exe.arg_dict['grid'][:] = test_grid
                exe.forward(is_train=True)
                exe.backward(mx.nd.array(out_grad))
                if req_dict['data'] is 'write':
                    assert_almost_equal(exe.grad_dict['data'], exe_list[ref_idx].grad_dict['data'], rtol=1e-3, atol=1e-5)
                if req_dict['grid'] is 'write':
                    assert_almost_equal(exe.grad_dict['grid'], exe_list[ref_idx].grad_dict['grid'], rtol=1e-3, atol=1e-5)



def test_correct1():
    data = nd.array([[[[1, 4, 3, 6],
                [1, 8, 8, 9],
                [0, 4, 1, 5],
                [1, 0, 1, 3]]]])

    affine_matrix = nd.array([[2, 0, 0],
      [0, 2, 0]])

    affine_matrix = nd.reshape(affine_matrix, shape=(1, 6))

    grid = nd.GridGenerator(data=affine_matrix,
    transform_type='affine', target_shape=(4, 4))

    out = nd.BilinearSampler(data, grid)
    print("out:\n", out)


def test_correct2():
    data= nd.array([[[[1, 4, 3, 6],
      [1, 8, 8, 9],
      [0, 4, 1, 5],
      [1, 0, 1, 3]]]])

    warp_matrix= nd.array([[[[1, 1, 1, 1],
      [1, 1, 1, 1],
      [1, 1, 1, 1],
      [1, 1, 1, 1]],
      [[0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0]]]])

    grid= nd.GridGenerator(data=warp_matrix, transform_type='warp')


    out= nd.BilinearSampler(data, grid)
    print("out:\n", out)



test_bilinear_sampler_with_type()
test_bilinear_sampler_versions()
test_correct1()
test_correct2()
