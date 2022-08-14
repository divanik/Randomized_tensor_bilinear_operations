import numpy as np
from bilinear_package.src import hadamard_product, contraction, primitives, convolution
from bilinear_package.src.random_tensor import createRandomTensor, createExampleTensor
from bilinear_package.src import rounding
import logging


def test_precise_convolution_running_correctness():
    for _ in range(100):
        modes = np.random.randint(10, size=10) + 2
        desired_ranks1 = np.random.randint(5, size=9) + 1
        random_tensor1 = createExampleTensor(
            modes, desired_ranks1, variance=10)
        desired_ranks2 = np.random.randint(5, size=9) + 1
        random_tensor2 = createExampleTensor(
            modes, desired_ranks2, variance=10)
        convolution.preciseCycleConvolution(
            random_tensor1, random_tensor2)


def test_precise_convolution_correctness():
    for _ in range(100):
        modes = [2, 1]
        desired_ranks1 = np.random.randint(5, size=1) + 1
        random_tensor1 = createExampleTensor(
            modes, desired_ranks1, variance=1)
        desired_ranks2 = np.random.randint(5, size=1) + 1
        random_tensor2 = createExampleTensor(
            modes, desired_ranks2, variance=1)
        x1, x3 = convolution.primitiveCycleConvolutionMatrices(
            random_tensor1, random_tensor2)
        x2 = primitives.countTensor(
            convolution.preciseCycleConvolution(random_tensor1, random_tensor2))
        print(x1)
        print(x2)
        print(x3)
        assert primitives.tensorsRelativeComparance(x1, x2) < 1e-10


def test_approximate_convolution_running_correctness():
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
        convolution.approximateCycleConvolution(
            random_tensor1, random_tensor2, random_tensor)


# def test_approximate_convolution_correctness():
#     for _ in range(10):
#         modes = np.random.randint(10, size=4) + 2
#         desired_ranks1 = np.random.randint(5, size=3) + 1
#         random_tensor1 = createExampleTensor(
#             modes, desired_ranks1, variance=1)
#         desired_ranks2 = np.random.randint(5, size=3) + 1
#         random_tensor2 = createExampleTensor(
#             modes, desired_ranks2, variance=1)
#         desired_ranks = np.random.randint(6, size=3) + 2
#         random_tensor = createRandomTensor(modes, desired_ranks)
#         product1 = convolution.approximateCycleConvolution(
#             random_tensor1, random_tensor2, random_tensor)
#         print(primitives.countTensor(product1).shape)
#         product2 = convolution.preciseCycleConvolution(
#             random_tensor1, random_tensor2)
#         print(primitives.countTensor(product2).shape)
#         kek = rounding.randomizeThenOrthogonalize(product2, random_tensor)
#         print(primitives.countTensor(kek).shape)
#         logging.warning(len(product1))
#         logging.warning(len(kek))
#         assert primitives.ttTensorsRelativeComparance(product1, kek) < 1e-4
