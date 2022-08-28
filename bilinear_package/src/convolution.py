from threading import main_thread
import typing
import numpy as np
from bilinear_package.src import contraction, primitives, hadamard_product
from bilinear_package.src.random_tensor_generation import createRandomTensor


def countFourier(tt_tensor: typing.List[np.ndarray]):
    answer = []
    for i in range(len(tt_tensor)):
        answer.append(np.fft.fft(tt_tensor[i], axis=1))
    return answer


def countInverseFourier(tt_tensor: typing.List[np.ndarray]):
    answer = []
    for i in range(len(tt_tensor)):
        answer.append(np.fft.ifft(tt_tensor[i], axis=1))
    return answer


def primitiveCycleConvolutionMatrices(tt_tensor1: typing.List[np.ndarray], tt_tensor2: typing.List[np.ndarray]):
    assert len(tt_tensor1) == 2
    assert len(tt_tensor2) == 2
    matrix1 = np.einsum(
        'ijk,klm->ijlm', tt_tensor1[0], tt_tensor1[1])[0, :, :, 0]
    matrix2 = np.einsum(
        'ijk,klm->ijlm', tt_tensor2[0], tt_tensor2[1])[0, :, :, 0]
    answer = np.zeros(matrix1.shape)
    assert matrix1.shape == matrix2.shape
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            kek = np.roll(np.flip(matrix2), (i + 1, j + 1), axis=(0, 1))
            answer[i, j] = np.sum(matrix1 * kek)
    return answer


def preciseCycleConvolution(tt_tensor1: typing.List[np.ndarray], tt_tensor2: typing.List[np.ndarray]):
    return countInverseFourier(hadamard_product.preciseHadamardProduct(countFourier(tt_tensor1), countFourier(tt_tensor2)))


def approximateCycleConvolution(tt_tensors1: typing.List[np.ndarray], tt_tensors2: typing.List[np.ndarray], desired_ranks: typing.List[int], seed: int):
    modes = primitives.countModes(tt_tensors1)
    random_tensor = createRandomTensor(modes, desired_ranks, seed)
    return hadamard_product.generalizedApproximateHadamardProduct(countFourier(tt_tensors1), countFourier(tt_tensors2), countInverseFourier(random_tensor), lambda z: np.fft.ifft(z, axis=2))
