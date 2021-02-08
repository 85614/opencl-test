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

def with_seed(*args, **kwargs):
  def decorator(func):
    return func
  return decorator

@with_seed()
def test_spatial_transformer_with_type():
    data = mx.sym.Variable('data')
    loc = mx.sym.Flatten(data)
    loc = mx.sym.FullyConnected(data=loc, num_hidden=10)
    loc = mx.sym.Activation(data=loc, act_type='relu')
    loc = mx.sym.FullyConnected(data=loc, num_hidden=6)
    sym = mx.sym.SpatialTransformer(data=data, loc=loc, target_shape=(10, 10),
                                    transform_type="affine", sampler_type="bilinear", cudnn_off=True)
    ctx_list = [{'ctx': mx.cpu(0), 'data': (1, 5, 10, 10), 'type_dict': {'data': np.float64}},
                {'ctx': mx.cpu(0), 'data': (1, 5, 10, 10), 'type_dict': {'data': np.float64}}]
    check_consistency(sym, ctx_list)
    check_consistency(sym, ctx_list, grad_req="add")
    sym = mx.sym.SpatialTransformer(data=data, loc=loc, target_shape=(10, 10),
                                    transform_type="affine", sampler_type="bilinear", cudnn_off=False)
    check_consistency(sym, ctx_list)
    check_consistency(sym, ctx_list, grad_req="add")


def my_test():
    test_spatial_transformer_with_type()

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
