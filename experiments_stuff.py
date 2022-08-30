from bilinear_package.src import random_tensor_generation
from collections import defaultdict
import pandas as pd
import time


def get_tensor_for_experiment(d: int, n: int, r: int, variance: float, seed: int):
    modes = [n for _ in range(d)]
    ranks = [r for _ in range(d - 1)]
    return random_tensor_generation.createExampleTensor(modes, ranks, variance=variance, seed=seed)


def get_random_tensor_for_experiment(d: int, n: int, r: int, seed: int):
    modes = [n for _ in range(d)]
    ranks = [n for _ in range(d - 1)]
    return random_tensor_generation.createRandomTensor(modes, ranks, seed=seed)


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


def count_time(default_config, parameter_name_set, grid_set, function, save_path):
    def make_benching(function, config):
        answer = {}
        for attempt in range(config["tries"]):
            tt_tensors1 = get_tensor_for_experiment(
                config["kernels"], config["modes"], config["initial_rank"], 1, attempt * 3)
            tt_tensors2 = get_tensor_for_experiment(
                config["kernels"], config["modes"], config["initial_rank"], 1, attempt * 3 + 1)
            desired_ranks = [config["rounding_rank"]
                             for _ in range(config["kernels"] - 1)]
            bench = time.time()
            kwargs = {
                "tt_tensors1": tt_tensors1,
                "tt_tensors2": tt_tensors2,
                "desired_ranks": desired_ranks,
                "seed": 3 * attempt + 2
            }
            function(**kwargs)
            answer[attempt] = time.time() - bench
        return answer

    config = default_config.copy()
    answer = dict()
    for parameter_set in grid_set:
        for i, name in enumerate(parameter_name_set):
            config[name] = parameter_set[i]
        answer[parameter_set] = make_benching(function, config)
    df = pd.DataFrame(answer)
    df.to_csv(save_path)
    return df
