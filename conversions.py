import numpy as np
import ast

def __MATRIX_FROM_FILE(filepath):
    with open(filepath, 'r') as F:
        STR = F.read()
        PY_LIST = ast.literal_eval(STR)
        MATRIX = np.matrix(PY_LIST)
        return MATRIX

def __ADJACENCY_TO_LAPLACIAN(A):
    L = -A
    for i in range(len(A)):
        DEG_i = 0
        for j in range(len(A)):
            DEG_i += A[i,j]
        L[i,i] = DEG_i
    return L

def __LAPLACIAN_TO_ADJACENCY(L):
    A = -L
    for i in range(len(A)):
        A[i,i] = 0
    return A

