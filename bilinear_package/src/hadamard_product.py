import typing
import logging
import numpy as np
from bilinear_package.src.contraction import cronMulVecL, partialContractionsRLKronecker
from bilinear_package.src.random_tensor import createRandomTensor


def preciseHadamardProduct(tt_tensors1: typing.List[np.array], tt_tensors2: typing.List[np.array]):
    answer = []
    for i in range(len(tt_tensors1)):
        a, b = tt_tensors1[i], tt_tensors2[i]
        result_kernel = np.zeros(
            (a.shape[0] * b.shape[0], a.shape[1], a.shape[2] * b.shape[2]), dtype=np.complex64)
        for i in range(a.shape[1]):
            result_kernel[:, i, :] = np.kron(a[:, i, :], b[:, i, :])
        answer.append(result_kernel)
    return answer


def approximateHadamardProduct(tt_tensors1: typing.List[np.array], tt_tensors2: typing.List[np.array], random_tensor: np.array):
    answer = []
    modes = []
    for tensor in tt_tensors1:
        modes.append(tensor.shape[1])
    contractions = partialContractionsRLKronecker(
        tt_tensors1L=tt_tensors1, tt_tensors1R=tt_tensors2, tt_tensors2=random_tensor)
    luls = np.ones((1, 1, 1))
    size = len(tt_tensors1)

    for i in range(size):
        z = cronMulVecL(tt_tensors1[i], tt_tensors2[i], luls)
        if i == size - 1:
            ans = np.transpose(z, (3, 2, 1, 0))
            ans = np.reshape(ans, (ans.shape[0], ans.shape[1], 1))
            answer.append(ans)
            return answer
        full = np.einsum('abcd,abe->dce', z, contractions[i])
        # logging.warning(z.shape)
        # logging.warning(contractions[i].shape)
        # logging.warning(full.shape)

        y = full.reshape((-1, full.shape[-1]), order='F')
        q, _ = np.linalg.qr(y)
        ans = q.reshape(full.shape[:-1] + (q.shape[1],), order='F')
        answer.append(ans)
        luls = np.einsum('abc,deba->dec', ans, z)

        # luls = cronMulVecReduceModeL(tt_tensors1[i], tt_tensors2[i], luls)
        # luls = luls.reshape(luls.shape[:2] + (-1,))
        # #logging.warning(luls.shape)
        # kek = q.reshape(
        #     (desired_ranks[i], tt_tensors2[i].shape[1]), order='F')
        # #logging.warning(kek.shape)
        # luls = np.einsum('abc,bcde->ade', kek, luls)
    return answer
