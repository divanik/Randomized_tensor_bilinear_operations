import typing
import numpy as np
from bilinear_package.src import orthogonalize, contraction, primitives
from bilinear_package.src.random_tensor_generation import createRandomTensor


def ttRoundingWithRanks(tt_tensors: typing.List[np.array], desired_ranks: typing.List[int]):
    answer = orthogonalize.orthogonalizeRL(tt_tensors)
    # print(answer[0].shape)
    # print(answer[1].shape)
    # print(answer[2].shape)
    for idx in range(len(answer) - 1):
        tensor = answer[idx]
        shape = tensor.shape
        y = primitives.makeVerticalUnfolding(tensor)
        y, r = np.linalg.qr(y)
        shape = (tensor.shape[0], tensor.shape[1], y.shape[1])
        answer[idx] = primitives.fromVerticalUnfolding(y, shape)

        # print(tensor.shape)
        # print(y.shape)
        # print(r.shape)
        # print(answer[idx].shape)
        # print(r.shape)
        U, S, Vt = np.linalg.svd(r, full_matrices=False)
        l = desired_ranks[idx]
        #print(l, U.shape, S.shape, Vt.shape)
        if l < S.size:
            U = U[:, :l]
            S = S[:l]
            Vt = Vt[:l, :]
        #print(answer[idx].shape, U.shape)
        answer[idx] = np.einsum('ijk,kl->ijl', answer[idx], U)
        #print(l, U.shape, S.shape, Vt.shape)

        SVt = np.diag(S) @ Vt
        #print(SVt.shape, answer[idx + 1].shape)
        answer[idx + 1] = np.einsum('ij,jkl->ikl', SVt, answer[idx + 1])
    return answer


def orthogonalizeThenRandomize(tt_tensors: typing.List[np.array], desired_ranks: typing.List[int]):
    answer = orthogonalize.orthogonalizeRL(tt_tensors)
    #print(f'Q {answer}')
    for idx in range(len(answer) - 1):
        tensor = answer[idx]
        shape = tensor.shape
        z = primitives.makeVerticalUnfolding(tensor)
        #print(f'z {z}')
        omega = np.random.normal(loc=0.0, scale=1/(z.shape[1] * desired_ranks[idx]), size=(
            z.shape[1], desired_ranks[idx]))  # there is a question about scale
        #print(f'Omega: {omega}')
        y = z @ omega
        #print(f'y {y}')
        v, _ = np.linalg.qr(y)
        answer[idx] = primitives.fromVerticalUnfolding(
            v, (shape[0], shape[1], v.shape[1]))
        #print(f'v {v}')
        m = v.T @ z
        #print(f'm {m}')
        # print()
        answer[idx + 1] = np.einsum('ij,jkl->ikl', m, answer[idx + 1])
    return answer


def randomizeThenOrthogonalize(tt_tensors: typing.List[np.array], random_tensor: typing.List[int]):
    contractions = contraction.partialContractionsRL(tt_tensors, random_tensor)
    answer = tt_tensors.copy()
    print(contractions)
    for idx in range(len(tt_tensors) - 1):
        z = primitives.makeVerticalUnfolding(answer[idx])
        shape = answer[idx].shape
        y = z @ contractions[idx]
        print(z)
        print("Y: ", y)
        y, _ = np.linalg.qr(y)
        print("Q: ", y)
        answer[idx] = primitives.fromVerticalUnfolding(
            y, (shape[0], shape[1], y.shape[1]))
        m = y.T @ z
        print("M: ", m)
        answer[idx + 1] = np.einsum('ij, jkl->ikl', m, answer[idx + 1])
    return answer


'''
def twoSidedRandomization(tt_tensors : typing.List[np.array], 
            desired_ranks : typing.List[int], helping_ranks : typing.List[int]):
    modes = []
    for tt in tt_tensors:
        modes.append(tt.shape[1])
    small_random_tensor = createRandomTensor(modes, desired_ranks)
    big_random_tensor = createRandomTensor(modes, helping_ranks)
    left_contractions = contraction.partialContractionsRL(tt_tensors, small_random_tensor)
    right_contractions = contraction.partialContractionsRL(tt_tensors, big_random_tensor)
    for idx in len(tt_tensors) - 1:
        U, S, Vt = np.linalg.svd(left_contractions[idx] @ right_contractions[idx])
        z = orthogonalize.makeVerticalUnfolding(answer[idx])
        shape = answer[idx].shape
        y = z * contractions[idx]
        y, _ = np.linalg.qr(y)
        answer[idx] = orthogonalize.fromVerticalUnfolding(y, (shape[0], shape[1], y.shape[1]))
        m = y.T @ z
        answer[idx + 1] = np.einsum('ij, jkl->ikl', m, answer[idx + 1])
'''
