import mxnet as mx
import mxnet.ndarray as nd
data = nd.array([[[[1, 4, 3, 6],
                [1, 8, 8, 9],
                [0, 4, 1, 5],
                [1, 0, 1, 3]]]])

affine_matrix = nd.array([[2, 0, 0],
                       [0, 2, 0]])

affine_matrix = nd.reshape(affine_matrix, shape=(1, 6))
grid = nd.GridGenerator(data=affine_matrix, transform_type='affine', target_shape=(4, 4))
affine_matrix.attach_grad()
grid.attach_grad()
with mx.autograd.record():
    out = nd.BilinearSampler(data, grid)
out.backward()
print(out)