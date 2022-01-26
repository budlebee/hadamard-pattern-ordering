# %%
import random
import numpy as np
from math import log2, floor
from scipy.linalg import hadamard


def make_hadamard_matrix(size):
    # N, N/12, N/20 이 2의 배수인지 체크해야함.
    # print(log2(size)-floor(log2(size)))
    # print(log2(size/12.0)-floor(log2(size/12)))
    # print(log2(size/20.0)-floor(log2(size/20.0)))
    n = floor(log2(size))
    if n-floor(n) != 0:
        print("size is not 2^n")
        return None
    print("size: ", size)
    H2 = np.array([[1, 1], [1, -1]])
    H = np.array([[1, 1], [1, -1]])
    for i in range(n-1):
        H = np.kron(H, H2)
    return np.where(H == -1, 0, H)


#H = make_hadamard_matrix(size=8)
H = hadamard(4)
print(H)

# %%
