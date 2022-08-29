import numpy as np
from bilinear_package.src import hadamard_product, contraction, primitives
from bilinear_package.src.random_tensor_generation import createRandomTensor, createExampleTensor
from bilinear_package.src import rounding
import logging


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
                kek, matricesRL1[i]) < 1e-5


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
        assert primitives.ttTensorsRelativeComparance(product1, kek) < 1e-4
