
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


def test_psroipooling():
    for num_rois in [1, 2]:
        for num_classes, num_group in itertools.product([2, 3], [2, 3]):
            for image_height, image_width in itertools.product([168, 224], [168, 224]):
                for grad_nodes in [['im_data']]:
                    spatial_scale = 0.0625
                    feat_height = np.int(image_height * spatial_scale)
                    feat_width = np.int(image_width * spatial_scale)
                    im_data = np.random.rand(1, num_classes*num_group*num_group, feat_height, feat_width)
                    rois_data = np.zeros([num_rois, 5])
                    rois_data[:, [1,3]] = np.sort(np.random.rand(num_rois, 2)*(image_width-1))
                    rois_data[:, [2,4]] = np.sort(np.random.rand(num_rois, 2)*(image_height-1))

                    im_data_var = mx.symbol.Variable(name="im_data")
                    rois_data_var = mx.symbol.Variable(name="rois_data")
                    op = mx.sym.contrib.PSROIPooling(data=im_data_var, rois=rois_data_var, spatial_scale=spatial_scale,
                                                     group_size=num_group, pooled_size=num_group,
                                                     output_dim=num_classes, name='test_op')
                    rtol, atol = 1e-2, 1e-3
                    check_numeric_gradient(op, [im_data, rois_data], rtol=rtol, atol=atol,
                                           grad_nodes=grad_nodes)
                    print('correct')




def my_test():        
    test_psroipooling()



if __name__ == '__main__':
    my_test()
    
        
    
    