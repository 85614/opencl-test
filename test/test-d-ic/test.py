import numbers
import sys
import os
import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.base import py_str, MXNetError, _as_list
from mxnet.test_utils import set_default_context, assert_almost_equal, assert_allclose, check_numeric_gradient
from mxnet.symbol import Symbol
from pickle import load, dump



output_file = sys.argv[1] if len(sys.argv) > 1 else "output"
input_file = sys.argv[2] if len(sys.argv) > 2 else "input"

def get_tolerance(rtol, ctx):
    if 'atol' in ctx:
        return ctx['atol']
    if 'atol_mult' in ctx:
        return ctx['atol_mult'] * rtol
    return rtol

def check_consistency(sym, ctx_list, scale=1.0, grad_req='write',
                      arg_params=None, aux_params=None, tol=None,
                      raise_on_err=True, ground_truth=None, equal_nan=False,
                      use_uniform=False, rand_type=np.float64):
    if tol is None:
        tol = {np.dtype(np.float16): 1e-1,
               np.dtype(np.float32): 1e-3,
               np.dtype(np.float64): 1e-5,
               np.dtype(np.uint8): 0,
               np.dtype(np.int32): 0,
               np.dtype(np.int64): 0}
    elif isinstance(tol, numbers.Number):
        tol = {np.dtype(np.float16): tol,
               np.dtype(np.float32): tol,
               np.dtype(np.float64): tol,
               np.dtype(np.uint8): tol,
               np.dtype(np.int32): tol,
               np.dtype(np.int64): tol}

    assert len(ctx_list) > 1
    if isinstance(sym, Symbol):
        sym = [sym]*len(ctx_list)
    else:
        assert len(sym) == len(ctx_list)

    output_names = sym[0].list_outputs()
    arg_names = sym[0].list_arguments()
    exe_list = []
    for s, ctx in zip(sym, ctx_list):
        assert s.list_arguments() == arg_names
        assert s.list_outputs() == output_names
        exe_list.append(s.simple_bind(grad_req=grad_req, **ctx))

    arg_params = {} if arg_params is None else arg_params
    aux_params = {} if aux_params is None else aux_params
    for n, arr in exe_list[0].arg_dict.items():
        if n not in arg_params:
            if fin.readable():
                # 从前面的记录里读取
                arg_params[n] = load(fin)
            else:
                if use_uniform:
                    arg_params[n] = np.random.uniform(low=-0.92, high=0.92,
                                                    size=arr.shape).astype(rand_type)
                else:
                    arg_params[n] = np.random.normal(size=arr.shape,
                                                 scale=scale).astype(rand_type)
                # 记录下来
                dump(arg_params[n], fin)
    for n, arr in exe_list[0].aux_dict.items():
        if n not in aux_params:
            aux_params[n] = 0
    for exe in exe_list:
        for name, arr in exe.arg_dict.items():
            arr[:] = arg_params[name]
        for name, arr in exe.aux_dict.items():
            arr[:] = aux_params[name]
        # We need to initialize the gradient arrays if it's add.
        if (grad_req == "add"):
            for arr in exe.grad_arrays:
                arr[:] = np.zeros(arr.shape, dtype=arr.dtype)

    dtypes = [np.dtype(exe.outputs[0].dtype) for exe in exe_list]
    max_idx = np.argmax(dtypes)
    gt = ground_truth
    if gt is None:
        gt = exe_list[max_idx].output_dict.copy()
        if grad_req != 'null':
            gt.update(exe_list[max_idx].grad_dict)


    # test
    for exe in exe_list:
        exe.forward(is_train=False)

    for i, exe in enumerate(exe_list):
        if i == max_idx:
            continue

        rtol = tol[dtypes[i]]
        atol = get_tolerance(rtol, ctx_list[i])
        for name, arr in zip(output_names, exe.outputs):
            # Previously, the cast was to dtypes[i], but symbol may be mixed-precision,
            # so casting the ground truth to the actual output type seems more correct.
            gtarr = gt[name].astype(arr.dtype)
            try:
                if fout.readable():
                    assert_almost_equal(arr, load(fout), rtol=rtol, atol=atol, equal_nan=equal_nan)
                else:
                    dump(arr, fout)
                assert_almost_equal(arr, gtarr, rtol=rtol, atol=atol, equal_nan=equal_nan)
            except AssertionError as e:
                print('Predict Err: ctx %d vs ctx %d at %s'%(i, max_idx, name))
                traceback.print_exc()
                if raise_on_err:
                    raise e

                print(str(e))

    # train
    if grad_req != 'null':
        for exe in exe_list:
            exe.forward(is_train=True)
            exe.backward(exe.outputs)

        for i, exe in enumerate(exe_list):
            if i == max_idx:
                continue

            rtol = tol[dtypes[i]]
            atol = get_tolerance(rtol, ctx_list[i])
            curr = zip(output_names + arg_names, exe.outputs + exe.grad_arrays)
            for name, arr in curr:
                if gt[name] is None:
                    assert arr is None
                    continue

                # Previous cast was to dtypes[i], but symbol may be mixed-precision,
                # so casting the ground truth to the actual output type seems more correct.
                gtarr = gt[name].astype(arr.dtype)
                try:
                    if fout.readable():
                        assert_almost_equal(arr, load(fout), rtol=rtol, atol=atol, equal_nan=equal_nan)
                    else:
                        dump(arr, fout)
                    assert_almost_equal(arr, gtarr, rtol=rtol, atol=atol, equal_nan=equal_nan)
                except AssertionError as e:
                    print('Train Err: ctx %d vs ctx %d at %s'%(i, max_idx, name))
                    traceback.print_exc()
                    if raise_on_err:
                        raise e

                    print(str(e))

    return gt

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
def test_deformable_convolution_with_type():
    tol = {np.dtype(np.float32): 1e-1,
           np.dtype(np.float64): 1e-3}

    sym = mx.sym.contrib.DeformableConvolution(num_filter=3, kernel=(3,3), name='deformable_conv')
    # since atomicAdd does not support fp16 (which deformable conv uses in backward), we do not test fp16 here
    ctx_list = [{'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 10, 10),
                 'deformable_conv_offset': (2, 18, 8, 8),
                 'type_dict': {'deformable_conv_data': np.float64, 'deformable_conv_offset': np.float64}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 10, 10),
                 'deformable_conv_offset': (2, 18, 8, 8),
                 'type_dict': {'deformable_conv_data': np.float32, 'deformable_conv_offset': np.float32}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 10, 10),
                 'deformable_conv_offset': (2, 18, 8, 8),
                 'type_dict': {'deformable_conv_data': np.float64, 'deformable_conv_offset': np.float64}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 10, 10),
                 'deformable_conv_offset': (2, 18, 8, 8),
                 'type_dict': {'deformable_conv_data': np.float32, 'deformable_conv_offset': np.float32}},
                ]

    check_consistency(sym, ctx_list, scale=0.1, tol=tol)
    # test ability to turn off training on bias
    check_consistency(sym, ctx_list, scale=0.1, tol=tol,
                      grad_req={'deformable_conv_data': 'write',
                                'deformable_conv_offset': 'write',
                                'deformable_conv_weight': 'write',
                                'deformable_conv_bias': 'null'})


@with_seed()
def test_deformable_convolution_options():
    tol = {np.dtype(np.float32): 1e-1,
           np.dtype(np.float64): 1e-3}
    # 2D convolution
    # since atomicAdd does not support fp16 (which deformable conv uses in backward), we do not test fp16 here

    # Pad > 0
    ctx_list = [{'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 7, 7),
                 'type_dict': {'deformable_conv_data': np.float64, 'deformable_conv_offset': np.float64}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 7, 7),
                 'type_dict': {'deformable_conv_data': np.float32, 'deformable_conv_offset': np.float32}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 7, 7),
                 'type_dict': {'deformable_conv_data': np.float64, 'deformable_conv_offset': np.float64}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 7, 7),
                 'type_dict': {'deformable_conv_data': np.float32, 'deformable_conv_offset': np.float32}},
                ]
    sym = mx.sym.contrib.DeformableConvolution(num_filter=3, kernel=(3,3), pad=(1,1), name='deformable_conv')
    check_consistency(sym, ctx_list, scale=0.1, tol=tol)

    # Stride > 1
    ctx_list = [{'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 3, 3),
                 'type_dict': {'deformable_conv_data': np.float64, 'deformable_conv_offset': np.float64}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 3, 3),
                 'type_dict': {'deformable_conv_data': np.float32, 'deformable_conv_offset': np.float32}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 3, 3),
                 'type_dict': {'deformable_conv_data': np.float64, 'deformable_conv_offset': np.float64}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 3, 3),
                 'type_dict': {'deformable_conv_data': np.float32, 'deformable_conv_offset': np.float32}},
                ]
    sym = mx.sym.contrib.DeformableConvolution(num_filter=3, kernel=(3,3), stride=(2,2), name='deformable_conv')
    check_consistency(sym, ctx_list, scale=0.1, tol=tol)

    # Dilate > 1
    ctx_list = [{'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 3, 3),
                 'type_dict': {'deformable_conv_data': np.float64, 'deformable_conv_offset': np.float64}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 18, 3, 3),
                 'type_dict': {'deformable_conv_data': np.float32, 'deformable_conv_offset': np.float32}},
                ]
    sym = mx.sym.contrib.DeformableConvolution(num_filter=3, kernel=(3,3), dilate=(2,2), name='deformable_conv')
    check_consistency(sym, ctx_list, scale=0.1, tol=tol)

    # Deformable group > 1
    ctx_list = [{'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 36, 5, 5),
                 'type_dict': {'deformable_conv_data': np.float64, 'deformable_conv_offset': np.float64}},
                {'ctx': mx.cpu(0),
                 'deformable_conv_data': (2, 2, 7, 7),
                 'deformable_conv_offset': (2, 36, 5, 5),
                 'type_dict': {'deformable_conv_data': np.float32, 'deformable_conv_offset': np.float32}},
                ]
    sym = mx.sym.contrib.DeformableConvolution(num_filter=4, kernel=(3,3), num_deformable_group=2, name='deformable_conv')
    check_consistency(sym, ctx_list, scale=0.1, tol=tol)


if __name__ == '__main__':
    in_mode = "rb" if os.path.exists(input_file) else "wb"
    out_mode = "rb" if "r" in in_mode and os.path.exists(output_file) else "wb"
    print("{} input in file '{}'".format("use old" if "r" in in_mode else "store random", input_file))
    print("{} output in file '{}'".format("compare" if "r" in out_mode else "store", output_file))
    with open(input_file, in_mode) as fin, open(output_file, out_mode) as fout:
        test_deformable_convolution_options()
        test_deformable_convolution_with_type()
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
    
        
    
    