from bilinear_package.src import random_tensor
from collections import defaultdict
import pandas as pd


def get_tensor_for_experiment(n: int, d: int, r: int, variance: float):
    modes = [d for _ in range(n)]
    ranks = [r for _ in range(n - 1)]
    return random_tensor.createExampleTensor(modes, ranks, variance=variance)


def get_random_tensor_for_experiment(n: int, d: int, r: int):
    modes = [d for _ in range(n)]
    ranks = [r for _ in range(n - 1)]
    return random_tensor.createRandomTensor(modes, ranks)


def default_dict_to_seaborn_plot(results: defaultdict):
    xs = []
    ys = []
    for key, value in results.items():
        xs.extend([key for _ in range(len(value))])
        ys.extend(value)
    return xs, ys


def default_dict_to_df(results: defaultdict, save_path: str):
    results = dict(results)
    df = pd.DataFrame(results)
    df.to_csv(save_path)
