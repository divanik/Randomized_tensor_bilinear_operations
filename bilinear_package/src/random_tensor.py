import typing
import numpy as np
from bilinear_package.src import primitives


def createRandomTensor(modes: typing.List[np.int32], ranks: typing.List[np.int32]):
    answer = []
    for idx in range(len(modes)):
        l1 = 1 if idx == 0 else ranks[idx - 1]
        l2 = 1 if idx == len(modes) - 1 else ranks[idx]
        tensor = np.random.normal(
            loc=0.0, scale=1 / (l1 * modes[idx] * l2), size=(l1, modes[idx], l2))
        answer.append(tensor)
    return answer
