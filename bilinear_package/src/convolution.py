import typing
import numpy as np
from bilinear_package.src import contraction, primitives


def convolution(tensor1 : np.ndarray, tensor2 : np.ndarray):
    assert tensor1.shape == tensor2.shape
    