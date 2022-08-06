import numpy as np
from bilinear_package.src import bilinear_operations
from bilinear_package.src import primitives

t0 = np.array([[[4, 5, 6], [-4, 7, 8], [1, -4, 5]]])

t1 = np.array([[[2, 3, 4], [5, 2, 8]], [[1, -4, 1],
              [1, -2, 0]], [[6, -4, -3], [4, 2, -7]]])

t2 = np.array([[[-2], [9], [-1]], [[9], [-10], [1]], [[1], [4], [2]]])


def test_sum_rand_then_orth():
    tt = [t0, t1, t2]

    # print(t0.shape)
    # print(t1.shape)
    # print(t2.shape)

    # sum0 = primitives.countTensor(
    #     bilinear_operations.roundingSumRandThenOrth(tt, tt, [2, 2]))
    # sum1 = primitives.countTensor(tt) + primitives.countTensor(tt)

    # print(primitives.countTensor(tt))
    # print(primitives.frob(primitives.frob(sum0) - primitives.frob(sum1)) /
    #       np.sqrt(primitives.frob(sum0) * primitives.frob(sum1)))
