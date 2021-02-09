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

def _validate_sample_location(input_rois, input_offset, spatial_scale, pooled_w, pooled_h, sample_per_part, part_size, output_dim, num_classes, trans_std, feat_h, feat_w):
    num_rois = input_rois.shape[0]
    output_offset = input_offset.copy()
    # simulate deformable psroipooling forward function
    for roi_idx in range(num_rois):
        sub_rois = input_rois[roi_idx, :].astype(np.float32)
        img_idx, x0, y0, x1, y1 = int(sub_rois[0]), sub_rois[1], sub_rois[2], sub_rois[3], sub_rois[4]
        roi_start_w = round(x0) * spatial_scale - 0.5
        roi_start_h = round(y0) * spatial_scale - 0.5
        roi_end_w = round(x1 + 1) * spatial_scale - 0.5
        roi_end_h = round(y1 + 1) * spatial_scale - 0.5
        roi_w, roi_h = roi_end_w - roi_start_w, roi_end_h - roi_start_h
        bin_size_w, bin_size_h = roi_w / pooled_w, roi_h / pooled_h
        sub_bin_size_w, sub_bin_size_h = bin_size_w / sample_per_part, bin_size_h / sample_per_part
        for c_top in range(output_dim):
            channel_each_cls = output_dim / num_classes
            class_id = int(c_top / channel_each_cls)
            for ph in range(pooled_h):
                for pw in range(pooled_w):
                    part_h = int(math.floor(float(ph) / pooled_h * part_size))
                    part_w = int(math.floor(float(pw) / pooled_w * part_size))
                    trans_x = input_offset[roi_idx, class_id * 2, part_h, part_w] * trans_std
                    trans_y = input_offset[roi_idx, class_id * 2 + 1, part_h, part_w] * trans_std
                    bin_h_start, bin_w_start = ph * bin_size_h + roi_start_h, pw * bin_size_w + roi_start_w

                    need_check = True
                    while need_check:
                        pass_check = True
                        for ih in range(sample_per_part):
                            for iw in range(sample_per_part):
                                h = bin_h_start + trans_y * roi_h + ih * sub_bin_size_h
                                w = bin_w_start + trans_x * roi_w + iw * sub_bin_size_w

                                if w < -0.5 or w > feat_w - 0.5 or h < -0.5 or h > feat_h - 0.5:
                                    continue

                                w = min(max(w, 0.1), feat_w - 1.1)
                                h = min(max(h, 0.1), feat_h - 1.1)
                                # if the following condiiton holds, the sampling location is not differentiable
                                # therefore we need to re-do the sampling process
                                if h - math.floor(h) < 1e-3 or math.ceil(h) - h < 1e-3 or w - math.floor(w) < 1e-3 or math.ceil(w) - w < 1e-3:
                                    trans_x, trans_y = random.random() * trans_std, random.random() * trans_std
                                    pass_check = False
                                    break
                            if not pass_check:
                                break
                        if pass_check:
                            output_offset[roi_idx, class_id * 2 + 1, part_h, part_w] = trans_y / trans_std
                            output_offset[roi_idx, class_id * 2, part_h, part_w] = trans_x / trans_std
                            need_check = False

    return output_offset

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


def my_test():
    pass

if __name__ == '__main__':
    # in_mode = "rb" if os.path.exists(input_file) else "wb"
    in_mode = "wb"
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
    
        
    
    