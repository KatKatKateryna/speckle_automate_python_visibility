import numpy as np

def expm(A, order=100):
    I = np.eye(A.shape[0])
    result = I
    term = I
    for n in range(1, order + 1):
        term = np.dot(term, A) / n
        result += term
    return result
