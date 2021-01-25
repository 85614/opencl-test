from mxnet import nd

# Zoom out data two times

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




# v1 = [1, 2]
# v2 = [0, 1]
# a = mx.sym.Variable('a')
# b = mx.sym.Variable('b')
# b_stop_grad = mx.ndarray.op.stop_gradient(3 * b)
# loss = mx.ndarray.op.MakeLoss(b_stop_grad + a)

# executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))
# executor.forward(is_train=True, a=v1, b=v2)
# executor.outputs
# [ 1.  5.]

# executor.backward()
# executor.grad_arrays
# [ 0.  0.]
# [ 1.  1.]