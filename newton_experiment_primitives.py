import numpy as np
import tt


def create_exponential_grid(d, n, A, alpha=1):
    x = np.reshape(np.linspace(-A, A, n, endpoint=True), (1, n, 1))
    ones = np.reshape(np.ones(n), (1, n, 1))
    base = [ones for _ in range(n)]
    X = []
    for i in range(d):
        current = base.copy()
        current[i] = x.copy()
        X.append(tt.vector.from_list(current))
    c = tt.multifuncrs2(X, lambda x: np.exp(-alpha *
                        np.sqrt(np.sum(x * x, axis=1))), eps=1E-6, verb=0)
    return tt.vector.to_list(c)


def create_netonial_potential_grid(d, n, A):
    x = np.reshape(np.linspace(-A, A, n, endpoint=True), (1, n, 1))
    ones = np.reshape(np.ones(n), (1, n, 1))
    base = [ones for _ in range(n)]
    X = []
    for i in range(d):
        current = base.copy()
        current[i] = x.copy()
        X.append(tt.vector.from_list(current))
    c = tt.multifuncrs2(X, lambda x: np.power(
        np.sum(x * x, axis=1), (d - 2) / 2), eps=1E-6, verb=0)
    return tt.vector.to_list(c)
