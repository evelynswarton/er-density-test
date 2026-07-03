import numpy as np
import networkx as nx

def __QUAD_FORM_PRIMITIVE(A, i, j):
    return A[i,i] + A[j,j] - A[i,j] - A[j,i]

def __QUAD_FORM_MATRIX(M):
    n = len(M)
    __MTRX_OUT = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            __MTRX_OUT[i,j] = __QUAD_FORM_PRIMITIVE(M,i,j)
    return __MTRX_OUT

def __DIST_RESISTANCE(L, i, j):
    L_PI = np.linalg.pinv(L)
    return __QUAD_FORM_PRIMITIVE(L_PI, i, j)

def __DIST_BIHARMONIC(L, i, j):
    L_PI_SQ = np.square(np.linalg.pinv(L))
    return __QUAD_FORM_PRIMITIVE(L_PI_SQ, i, j)

def __DIST_TRIHARMONIC(L, i, j):
    L_PI_CU = np.linalg.matrix_power(np.linalg.pinv(L), 3)
    return __QUAD_FORM_PRIMITIVE(L_PI_CU, i, j)

def __DIST_QUADHARMONIC(L, i, j):
    L_PI_QU = np.linalg.matrix_power(np.linalg.pinv(L), 4)
    return __QUAD_FORM_PRIMITIVE(L_PI_QU, i, j)

def __DIST_K_HARMONIC(__K, L, i, j):
    L_PI_PO = np.linalg.matrix_power(np.linalg.pinv(L), __K)
    return __QUAD_FORM_PRIMITIVE(L_PI_PO, i, j)

def __ALL_PAIRS_SHORTEST_PATH(L):
    print('Computing all pairs shortest path...')
    A = -L
    for i in range(len(A)):
        A[i,i] = 0
    G = nx.from_numpy_array(A)
    return nx.floyd_warshall_numpy(G)

