
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


def test_multi_proposal_op():
    # paramters
    feature_stride = 16
    scales = (8, 16, 32)
    ratios = (0.5, 1, 2)
    rpn_pre_nms_top_n = 12000
    rpn_post_nms_top_n = 2000
    threshold = 0.7
    rpn_min_size = 16

    batch_size = 20
    feat_len = (1000 + 15) // 16
    H, W = feat_len, feat_len
    num_anchors = len(scales) * len(ratios)
    count_anchors = H * W * num_anchors

    '''
    cls_prob: (batch_size, 2 * num_anchors, H, W)
    bbox_pred: (batch_size, 4 * num_anchors, H, W)
    im_info: (batch_size, 3)
    '''

    cls_prob = mx.nd.empty((batch_size, 2 * num_anchors, H, W), dtype = np.float32)
    bbox_pred = mx.nd.empty((batch_size, 4 * num_anchors, H, W), dtype = np.float32)
    im_info = mx.nd.empty((batch_size, 3), dtype = np.float32)

    cls_prob = mx.nd.array(np.random.random(cls_prob.shape))
    bbox_pred = mx.nd.array(np.random.random(bbox_pred.shape))
    cls_prob = handle(cls_prob)
    bbox_pred = handle(bbox_pred)
    for i in range(batch_size):
        im_size = np.random.randint(100, feat_len * feature_stride, size = (2,))
        im_scale = np.random.randint(70, 100) / 100.0
        im_info[i, :] = [im_size[0], im_size[1], im_scale]

    im_info = handle(im_info)

    def get_sub(arr, i):
        new_shape = list(arr.shape)
        new_shape[0] = 1
        res = arr[i].reshape(new_shape)
        return res

    def check_forward(rpn_pre_nms_top_n, rpn_post_nms_top_n):
        single_proposal = []
        single_score = []
        for i in range(batch_size):
            rois, score = mx.nd.contrib.Proposal(
                    cls_prob = get_sub(cls_prob, i),
                    bbox_pred = get_sub(bbox_pred, i),
                    im_info = get_sub(im_info, i),
                    feature_stride = feature_stride,
                    scales = scales,
                    ratios = ratios,
                    rpn_pre_nms_top_n = rpn_pre_nms_top_n,
                    rpn_post_nms_top_n = rpn_post_nms_top_n,
                    threshold = threshold,
                    rpn_min_size = rpn_min_size, output_score = True)
            single_proposal.append(rois)
            single_score.append(score)

        multi_proposal, multi_score = mx.nd.contrib.MultiProposal(
                cls_prob = cls_prob,
                bbox_pred = bbox_pred,
                im_info = im_info,
                feature_stride = feature_stride,
                scales = scales,
                ratios = ratios,
                rpn_pre_nms_top_n = rpn_pre_nms_top_n,
                rpn_post_nms_top_n = rpn_post_nms_top_n,
                threshold = threshold,
                rpn_min_size = rpn_min_size, output_score = True)

        single_proposal = mx.nd.stack(*single_proposal).reshape(multi_proposal.shape)
        single_score = mx.nd.stack(*single_score).reshape(multi_score.shape)

        single_proposal_np = single_proposal.asnumpy()
        multi_proposal_np = multi_proposal.asnumpy()

        single_score_np = single_score.asnumpy()
        multi_score_np = multi_score.asnumpy()

        # check rois x1,y1,x2,y2
        try:
            # assert np.allclose(single_proposal_np[:, 1:], multi_proposal_np[:, 1:], rtol=1e-6, atol=1e-4)
            assert_almost_equal(single_proposal_np[:, 1:], multi_proposal_np[:, 1:], atol=1e-4)
        except:
            # print(f'rpn_pre_nms_top_n={rpn_pre_nms_top_n}')
            # print(f'rpn_post_nms_top_n={rpn_post_nms_top_n}')
            # print(f'cls_prob={cls_prob}')
            # print(f'bbox_pred={bbox_pred}')
            # print(f'im_info={im_info}')
            # print(single_proposal_np[:, 1:], multi_proposal_np[:, 1:])
            left, right = single_proposal_np[:, 1:], multi_proposal_np[:, 1:]
            for i in range(len(left)):
                for j in range(len(left[i])):
                    if left[i][j] > 500 and right[i][j] < 1e-3:
                        print((i, j, left[i][j], right[i][j],), end='')
            raise
        # check rois batch_idx
        for i in range(batch_size):
            start = i * rpn_post_nms_top_n
            end = start + rpn_post_nms_top_n
            assert (multi_proposal_np[start:end, 0] == i).all()
        # check score
        # assert np.allclose(single_score_np, multi_score_np)
        assert_almost_equal(single_score_np, multi_score_np)

    def check_backward(rpn_pre_nms_top_n, rpn_post_nms_top_n):

        im_info_sym = mx.sym.Variable('im_info')
        cls_prob_sym = mx.sym.Variable('cls_prob')
        bbox_pred_sym = mx.sym.Variable('bbox_pred')

        sym = mx.sym.contrib.MultiProposal(
                cls_prob = cls_prob_sym,
                bbox_pred = bbox_pred_sym,
                im_info = im_info_sym,
                feature_stride = feature_stride,
                scales = scales,
                ratios = ratios,
                rpn_pre_nms_top_n = rpn_pre_nms_top_n,
                rpn_post_nms_top_n = rpn_post_nms_top_n,
                threshold = threshold,
                rpn_min_size = rpn_min_size, output_score = False)

        location = [cls_prob.asnumpy(), bbox_pred.asnumpy(), im_info.asnumpy()]

        expected = [np.zeros_like(e) for e in location]

        out_grads = [np.ones((rpn_post_nms_top_n, 5))]

        check_symbolic_backward(sym, location, out_grads, expected)

    check_forward(rpn_pre_nms_top_n, rpn_post_nms_top_n)
    print('correct')
    check_forward(rpn_pre_nms_top_n, 1500)
    print('correct')
    check_forward(1000, 500)
    print('correct')
    check_backward(rpn_pre_nms_top_n, rpn_post_nms_top_n)
    print('correct')

file_name = sys.argv[1] if len(sys.argv) > 1 else "input"
mode = sys.argv[2] if len(sys.argv) > 2 else "rb" if len(sys.argv) > 1 else "wb"
fin = None

def my_test(times=1):
    global fin
    for i in range(times):
        with open(file_name, mode) as fin:
            print(i)
            test_multi_proposal_op()


def handle(data):
    if (fin.readable()):
        return load(fin)
    else:
        dump(data, fin)
        return data



if __name__ == '__main__':
    my_test()
    
        
    
    