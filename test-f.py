import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
input = mx.sym.Variable('input')
output = mx.sym.Variable('output')
bias = mx.sym.Variable('bias')
mod = mx.sym.FullyConnected(input,output,bias,2)
amod =mod.bind(ctx=mx.cpu(),args={'input':mx.nd.array([1,2,3,4]),'output':mx.nd.array([[0.1],[0.2]]),'bias':mx.nd.array([[1],[1]])})
amod.forward()
