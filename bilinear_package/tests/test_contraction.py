import numpy as np
from bilinear_package.src import contraction, hadamard_product, random_tensor

t0 = np.array([[[4, 5, 6], [-4, 7, 8], [1, -4, 5]]])

t1 = np.array([[[2, 3, 4], [5, 2, 8]], [[1, -4, 1],
              [1, -2, 0]], [[6, -4, -3], [4, 2, -7]]])

t2 = np.array([[[-2], [9], [-1]], [[9], [-10], [1]], [[1], [4], [2]]])

print(t0.shape)
print(t1.shape)
print(t2.shape)

tt = [t0, t1, t2]

tt2 = [t2, t1, t0]

print(contraction.partialContractionsRL(tt, tt))

print(contraction.partialContractionsLR(tt, tt))


def test_partial_contraction_running_correctness():
    for _ in range(100):
        modes = np.random.randint(10, size=10) + 2
        desired_ranks1 = np.random.randint(5, size=9) + 1
        random_tensor1 = random_tensor.createRandomTensor(
            modes, desired_ranks1)
        desired_ranks2 = np.random.randint(5, size=9) + 1
        random_tensor2 = random_tensor.createRandomTensor(
            modes, desired_ranks2)
        matricesRL = contraction.partialContractionsRL(
            random_tensor1, random_tensor2)
        matricesLR = contraction.partialContractionsLR(
            random_tensor1, random_tensor2)
        assert(len(matricesLR) == len(matricesRL))
        scalar_products = []
        for i in range(len(matricesLR)):
            scalar_products.append(
                np.sum(matricesLR[i] * matricesRL[i]))
        assert(max(scalar_products) - min(scalar_products) < 1e-12)