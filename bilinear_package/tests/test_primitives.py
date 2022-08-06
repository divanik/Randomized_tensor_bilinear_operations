import numpy as np
from bilinear_package.src import primitives

t0 = np.array([ [[4 , 5, 6], [-4, 7, 8], [1, -4, 5]] ])

t1 = np.array([[[2, 3, 4], [5, 2, 8]], [[1, -4, 1], [1, -2, 0]], [[6, -4, -3], [4, 2, -7]]])

t2 = np.array([ [[-2] ,[9], [-1]], [[9], [-10], [1]], [[1], [4], [2]] ])

print(t0.shape)
print(t1.shape)
print(t2.shape)

tt = [t0, t1, t2]

tt2 = [2 * t0, t1, t2]

def testFastFrobeniusDistance():
    assert(abs(primitives.countFrobeniusDistance(tt, tt)) < 1e-10)
    assert(abs(primitives.countFrobeniusDistance(tt, tt2) - primitives.countFrobeniusDistanceSlow(tt, tt2) ) < 1e-10)