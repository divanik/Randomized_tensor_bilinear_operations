import numpy as np
from bilinear_package.src import orthogonalize
from bilinear_package.src import primitives


def test_orthoganality():
    t0 = np.array([[[4, 5, 6], [-4, 7, 8], [1, -4, 5]]])  # Shape 1, 3, 3

    t1 = np.array([[[2, 3, 4], [5, 2, 8]], [[1, -4, 1], [1, -2, 0]],
                  [[6, -4, -3], [4, 2, -7]]])  # Shape 3, 2, 3

    t2 = np.array([[[-2], [9], [0]], [[9], [-10], [3]],
                  [[1], [4], [9]]])  # Shape 3, 3, 1

    tt = [t0, t1, t2]

    tt2 = orthogonalize.orthogonalizeRL(tt)

    U2 = primitives.makeHorizontalUnfolding(tt2[2])
    U1 = primitives.makeHorizontalUnfolding(tt2[1])
    assert(np.linalg.norm(np.eye(U2.shape[0]) - U2 @ U2.T, ord='fro') < 1e-10)
    assert(np.linalg.norm(np.eye(U1.shape[0]) - U1 @ U1.T, ord='fro') < 1e-10)


def test_correctness():
    t0 = np.array([[[4, 5, 6], [-4, 7, 8], [1, -4, 5]]])  # Shape 1, 3, 3

    t1 = np.array([[[2, 3, 4], [5, 2, 8]], [[1, -4, 1], [1, -2, 0]],
                  [[6, -4, -3], [4, 2, -7]]])  # Shape 3, 2, 3

    t2 = np.array([[[-2], [9], [0]], [[9], [-10], [3]],
                  [[1], [4], [9]]])  # Shape 3, 3, 1

    tt = [t0, t1, t2]

    tt2 = orthogonalize.orthogonalizeRL(tt)

    U2 = primitives.makeHorizontalUnfolding(tt2[2])
    U1 = primitives.makeHorizontalUnfolding(tt2[1])
    assert(primitives.frob(primitives.countTensor(
        tt) - primitives.countTensor(tt2)) < 1e-10)
