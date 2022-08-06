import typing
import numpy as np

def preciseHadamardProduct(tt_tensors1: typing.List[np.array], tt_tensors2: typing.List[np.array]):
    answer = []
    for i in range(len(tt_tensors1)):
        answer.append(np.einsum('azc,dzf->adzcf'), tt_tensors1[i], tt_tensors2[i])
    return answer

def preci