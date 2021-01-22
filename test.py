
# Zoom out data two times

data = array([[[[1, 4, 3, 6],
            [1, 8, 8, 9],
            [0, 4, 1, 5],
            [1, 0, 1, 3]]]])

affine_matrix = array([[2, 0, 0],
  [0, 2, 0]])

affine_matrix = reshape(affine_matrix, shape=(1, 6))

grid = GridGenerator(data=affine_matrix,
transform_type='affine', target_shape=(4, 4))

out = BilinearSampler(data, grid)

print("out:\n", out)
print("excepted:",
  [[[[0,   0,     0,   0],
  [0,   3.5,   6.5, 0],
  [0,   1.25,  2.5, 0],
  [0,   0,     0,   0]]]])



data= array([[[[1, 4, 3, 6],
  [1, 8, 8, 9],
  [0, 4, 1, 5],
  [1, 0, 1, 3]]]])

warp_maxtrix= array([[[[1, 1, 1, 1],
  [1, 1, 1, 1],
  [1, 1, 1, 1],
  [1, 1, 1, 1]],
  [[0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0]]]])

grid= GridGenerator(data=warp_matrix, transform_type='warp')
out= BilinearSampler(data, grid)

print("out:\n", out)
print("excepted:",
  [[[[4,  3,  6,  0],
  [8,  8,  9,  0],
  [4,  1,  5,  0],
  [0,  1,  3,  0]]]])
