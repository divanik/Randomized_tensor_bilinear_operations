import numpy as np
from bilinear_package.src import hadamard_product

t0 = np.array([[[4, 5, 6], [-4, 7, 8], [1, -4, 5]]])

t1 = np.array([[[2, 3, 4], [5, 2, 8]], [[1, -4, 1],
              [1, -2, 0]], [[6, -4, -3], [4, 2, -7]]])

t2 = np.array([[[-2], [9], [-1]], [[9], [-10], [1]], [[1], [4], [2]]])

print(t0.shape)
print(t1.shape)
print(t2.shape)

tt = [t0, t1, t2]

tt2 = [t2, t1, t0]


def testFastFrobeniusDistance():
    hadamard_product.preciseHadamardProduct(tt, tt2)
