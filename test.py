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
print("data:\n", data)
print("grid:\n", grid)

print("excepted:",
  [[[[0,   0,     0,   0],
  [0,   3.5,   6.5, 0],
  [0,   1.25,  2.5, 0],
  [0,   0,     0,   0]]]])
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

print("data:\n", data)
print("grid:\n", grid)
print("excepted:",
  [[[[4,  3,  6,  0],
  [8,  8,  9,  0],
  [4,  1,  5,  0],
  [0,  1,  3,  0]]]])
out= nd.BilinearSampler(data, grid)
print("out:\n", out)

