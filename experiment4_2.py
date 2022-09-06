import numpy as np
import tt
import experiments_stuff
from bilinear_package.src import primitives
from bilinear_package.src.hadamard_product import generalizedApproximateHadamardProduct
from bilinear_package.src.convolution import approximateConvolution, preciseConvolution, countFourier, countInverseFourier, approximateCycleConvolution
import time


from newton_experiment_primitives import create_exponential_grid, create_newtonial_potential_grid

from bilinear_package.src.convolution import preciseCycleConvolution
from newton_experiment_primitives import interpolateTTTensor, compressTTTensor
import pandas as pd

D = 40

df_relation = pd.DataFrame()

for d in range(3, 6):
    answer = []
    for grid_side in np.logspace(7, 13, 7, base = 2, dtype = int):
        h = (2 * D) / grid_side
        tensor1 = create_exponential_grid(d, grid_side + 2, D + h / 2)
        tensor2 = create_newtonial_potential_grid(d, 2 * grid_side + 2, h * grid_side + h / 2)
        tensor1 = primitives.twoSidedPaddingTTTensor(tensor1, [(0, grid_side) for _ in range(d)])
        approx = approximateCycleConvolution(tensor1, tensor2, desired_ranks=np.ones(d - 1, dtype=int) * 30, seed = 271 * d + grid_side)
        approx = primitives.twoSidedCuttingTTTensor(approx, [(0, grid_side + 1) for _ in range(d)])
        for i in range(len(approx)):
            approx[i] = approx[i] * h
        answer.append(approx)
        print(grid_side)
    writer = []
    for mid in range(1, 6):
        A = answer[mid - 1].copy()
        B = answer[mid].copy()
        C = answer[mid + 1].copy()
        A = tt.vector.from_list(interpolateTTTensor(A))
        B = tt.vector.from_list(B)
        C = tt.vector.from_list(compressTTTensor(C))
        writer.append(tt.vector.norm(B - C) / tt.vector.norm(A - B))
    df_relation[d] = writer
    df_relation.to_csv("experiments_results/experiment4/relation.csv")