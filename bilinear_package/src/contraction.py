import typing
import logging
import numpy as np


def partialContractionsRL(tt_tensors1: typing.List[np.array], tt_tensors2: typing.List[np.array]):
    answer = []
    last = np.ones((1, 1))
    for idx, tt1, tt2 in (zip(reversed(range(len(tt_tensors1))), reversed(tt_tensors1), reversed(tt_tensors2))):
        if idx > 0:
            last = np.einsum('ijk,kl,mjl->im', tt1, last, tt2)
            answer.append(last.copy())
    return list(reversed(answer))


def partialContractionsLR(tt_tensors1: typing.List[np.array], tt_tensors2: typing.List[np.array]):
    answer = []
    cur = np.ones((1, 1))
    for idx, tt1, tt2 in zip(range(len(tt_tensors1)), tt_tensors1, tt_tensors2):
        if idx < (len(tt_tensors1) - 1):
            cur = np.einsum('ijk,il,ljm->km', tt1, cur, tt2)
            answer.append(cur.copy())
    return answer


def cronMulVecR(a: np.array, b: np.array, c: np.array):
    p = np.einsum('dbe,cel->dbcl', b, c)
    return np.einsum('abc,dbcl->adbl', a, p)


def cronMulVecL(a: np.array, b: np.array, c: np.array):
    p = np.einsum('ebd,cel->dbcl', b, c)
    return np.einsum('cba,dbcl->adbl', a, p)


def cronMulVecReduceModeR(a: np.array, b: np.array, c: np.array):
    p = np.einsum('dbe,cebl->dcbl', b, c)
    return np.einsum('abc,dcbl->adl', a, p)


def cronMulVecReduceModeL(a: np.array, b: np.array, c: np.array):
    p = np.einsum('ebd,cebl->cdbl', b, c)
    return np.einsum('cba,cdbl->adl', a, p)


def partialContractionsRLKronecker(tt_tensors1L: typing.List[np.array], tt_tensors1R: typing.List[np.array], tt_tensors2: typing.List[np.array]):
    answer = []
    last = np.ones((1, 1, 1))
    for idx, tt1L, tt1R, tt2 in (zip(reversed(range(len(tt_tensors1L))), reversed(tt_tensors1L), reversed(tt_tensors1R), reversed(tt_tensors2))):
        p = cronMulVecR(tt1L, tt1R, last)
        last = np.einsum('ldu,abdu->abl', tt2, p)
        answer.append(last.copy())
    return list(reversed(answer))
