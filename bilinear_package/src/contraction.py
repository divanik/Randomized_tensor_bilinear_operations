import typing
import numpy as np


def partialContractionsRL(tt_tensors1: typing.List[np.array], tt_tensors2: typing.List[np.array]):
    answer = []
    last = np.ones((1, 1))
    for idx, tt1, tt2 in (zip(reversed(range(len(tt_tensors1))), reversed(tt_tensors1), reversed(tt_tensors2))):
        if idx > 0:
            # print(idx)
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


def partialContractionsRLKronecker(tt_tensors1L: typing.List[np.array], tt_tensors1R: typing.List[np.array], tt_tensors2: typing.List[np.array]):
    answer = []
    last = np.ones((1, 1, 1))
    for idx, tt1L, tt1R, tt2 in (zip(reversed(range(len(tt_tensors1L))), reversed(tt_tensors1L), reversed(tt_tensors1R), reversed(tt_tensors2))):
        if idx > 0:
            p = np.einsum('lzk,bdk->bdzk', tt2, last)
            p = np.einsum('czd,bdzk->bczk', tt1R, p)
            last = np.einsum('azb,bczk->ack', tt1L, p)
            answer.append(last.copy())
    return list(reversed(answer))
