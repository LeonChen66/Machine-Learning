import numpy as np
from scipy import signal



I = np.array([[164,188,164,161,195],[178,201,197,150,137],
    [174,168,181,190,184],[131,179,176,185,198],[92,185,179,133,167]])
F1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
F2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
F3 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

f1 = signal.convolve2d(I, F1, 'valid')
f2 = signal.convolve2d(I, F2, 'valid')
f3 = signal.convolve2d(I, F3, 'valid')

print(f1,f2,f3)



def convolve2d_scratch(a, conv_filter):
    submatrices = np.array([
        [a[:-2, :-2], a[:-2, 1:-1], a[:-2, 2:]],
        [a[1:-1, :-2], a[1:-1, 1:-1], a[1:-1, 2:]],
        [a[2:, :-2], a[2:, 1:-1], a[2:, 2:]]])
    multiplied_subs = np.einsum('ij,ijkl->ijkl', conv_filter, submatrices)
    return np.sum(np.sum(multiplied_subs, axis=-3), axis=-3)
