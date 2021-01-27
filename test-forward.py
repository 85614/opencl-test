from mxnet import nd


def test_correct1():
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


def test_correct2():
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


