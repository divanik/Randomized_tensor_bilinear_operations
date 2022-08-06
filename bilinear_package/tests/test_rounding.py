import numpy as np
from bilinear_package.src.primitives import countTensor, frob
from bilinear_package.src import rounding


t0 = np.array([ [[4 , 5, 6], [-4, 7, 8], [1, -4, 5]] ])

t1 = np.array([[[2, 3, 4], [5, 2, 8]], [[1, -4, 1], [1, -2, 0]], [[6, -4, -3], [4, 2, -7]]])

t2 = np.array([ [[-2] ,[9], [-1]], [[9], [-10], [1]], [[1], [4], [2]] ])

tt = [t0, t1, t2]

#-------------------------------------------------------

def testOrthogonalizeThenRandomize():
    #def testTrivialRounging():
    tt2 = rounding.orthogonalizeThenRandomize(tt, (3, 3))
        
    assert(frob(countTensor(tt) - countTensor(tt2)) / np.sqrt(frob(countTensor(tt)) * frob(countTensor(tt2))) < 1e-10) 

#-------------------------------------------------------

def testRoundingCorrectness():
    for i in range(1, 4):
        for j in range(1, 4):
            tt2 = rounding.orthogonalizeThenRandomize(tt,  [i, j])
        assert( frob(countTensor(tt) - countTensor(tt2)) / np.sqrt(frob(countTensor(tt)) * frob(countTensor(tt2))) )

#--------------------------------------------------------