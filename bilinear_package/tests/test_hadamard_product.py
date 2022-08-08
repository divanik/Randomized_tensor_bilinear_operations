from itertools import count
from statistics import variance
import numpy as np
from bilinear_package.src import hadamard_product, contraction, primitives
from bilinear_package.src.random_tensor import createRandomTensor, createExampleTensor
from bilinear_package.src import rounding
import logging

# t0 = np.array([[[4, 5, 6], [-4, 7, 8], [1, -4, 5]]])

# t1 = np.array([[[2, 3, 4], [5, 2, 8]], [[1, -4, 1],
#               [1, -2, 0]], [[6, -4, -3], [4, 2, -7]]])

# t2 = np.array([[[-2], [9], [-1]], [[9], [-10], [1]], [[1], [4], [2]]])

# print(t0.shape)
# print(t1.shape)
# print(t2.shape)

# tt = [t0, t1, t2]

# tt2 = [t2, t1, t0]


def test_precise_hadamard_product_running_correctness():
    for _ in range(100):
        modes = np.random.randint(10, size=10) + 2
        desired_ranks1 = np.random.randint(5, size=9) + 1
        random_tensor1 = list(map(lambda x: x * 1000, createRandomTensor(
            modes, desired_ranks1)))
        desired_ranks2 = np.random.randint(5, size=9) + 1
        random_tensor2 = list(map(lambda x: x * 1000, createRandomTensor(
            modes, desired_ranks2)))
        hadamard_product.preciseHadamardProduct(random_tensor1, random_tensor2)


def test_precise_hadamard_product_correctness():
    for _ in range(1000):
        modes = np.random.randint(10, size=3) + 2
        desired_ranks1 = np.random.randint(5, size=2) + 1
        random_tensor1 = createExampleTensor(
            modes, desired_ranks1, variance=100)
        desired_ranks2 = np.random.randint(5, size=2) + 1
        random_tensor2 = createExampleTensor(
            modes, desired_ranks2, variance=100)
        honest_hadamard_product = primitives.countTensor(
            random_tensor1) * primitives.countTensor(random_tensor2)
        assert primitives.tensorsRelativeComparance(
            primitives.countTensor(hadamard_product.preciseHadamardProduct(random_tensor1, random_tensor2)), honest_hadamard_product)


def test_partial_contraction_kronecker():
    for _ in range(100):
        modes = np.random.randint(10, size=10) + 2
        desired_ranks1 = np.random.randint(5, size=9) + 1
        random_tensor1 = createExampleTensor(
            modes, desired_ranks1, 100)
        desired_ranks2 = np.random.randint(5, size=9) + 1
        random_tensor2 = createExampleTensor(
            modes, desired_ranks2, 100)
        desired_ranks = np.random.randint(6, size=9) + 2
        random_tensor = createExampleTensor(
            modes, desired_ranks, 100)
        hadamard_product_tensor = hadamard_product.preciseHadamardProduct(
            random_tensor1, random_tensor2)
        matricesRL1 = contraction.partialContractionsRL(
            hadamard_product_tensor, random_tensor)
        matricesRL2 = contraction.partialContractionsRLKronecker(
            random_tensor1, random_tensor2, random_tensor)
        assert len(matricesRL1) == len(matricesRL2)
        size = len(matricesRL1)
        for i in range(size):
            kek = matricesRL2[i].reshape(
                (-1, matricesRL2[i].shape[2]), order='C')
            assert primitives.tensorsRelativeComparance(
                kek, matricesRL1[i]) < 1e-10


def test_approximate_hadamard_product_running_correctness():
    for _ in range(100):
        modes = np.random.randint(10, size=10) + 2
        desired_ranks1 = np.random.randint(5, size=9) + 1
        random_tensor1 = createExampleTensor(
            modes, desired_ranks1, variance=100)
        desired_ranks2 = np.random.randint(5, size=9) + 1
        random_tensor2 = createExampleTensor(
            modes, desired_ranks2, variance=100)
        desired_ranks = np.random.randint(6, size=9) + 2
        random_tensor = createRandomTensor(modes, desired_ranks)
        hadamard_product.approximateHadamardProduct(
            random_tensor1, random_tensor2, random_tensor)


def test_approximate_hadamard_product_correctness():
    for _ in range(10):
        modes = np.random.randint(10, size=4) + 2
        desired_ranks1 = np.random.randint(5, size=3) + 1
        random_tensor1 = createExampleTensor(
            modes, desired_ranks1, variance=1)
        desired_ranks2 = np.random.randint(5, size=3) + 1
        random_tensor2 = createExampleTensor(
            modes, desired_ranks2, variance=1)
        desired_ranks = np.random.randint(6, size=3) + 2
        random_tensor = createRandomTensor(modes, desired_ranks)
        product1 = hadamard_product.approximateHadamardProduct(
            random_tensor1, random_tensor2, random_tensor)
        print(primitives.countTensor(product1).shape)
        product2 = hadamard_product.preciseHadamardProduct(
            random_tensor1, random_tensor2)
        print(primitives.countTensor(product2).shape)
        kek = rounding.randomizeThenOrthogonalize(product2, random_tensor)
        print(primitives.countTensor(kek).shape)
        logging.warning(len(product1))
        logging.warning(len(kek))
        assert primitives.ttTensorsRelativeComparance(product1, kek) < 1
