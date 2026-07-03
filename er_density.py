import os
import sys
import string
import json
import math
import numpy as np

GRAPHS_DIR = './graphs/'
F_ADJ_MAT  = 'adjacency_matrix.json'
F_RES_MAT  = 'resistance_matrix.json'
F_RES_MLT  = 'resistance_multiset.json'
F_RES_DNS  = 'resistance_density.json'
F_RES_GRO  = 'resistance_growth.json'

def all_graph_dirs() -> list[os.path]:
    graph_dirs = []
    for root, dirs, files in os.walk(GRAPHS_DIR):
        for name in dirs:
            if name not in ['graphs', 'planar', 'delaunay']:
                graph_dirs.append(os.path.join(root, name))
    return graph_dirs

def er_growth(resistance_matrix: np.ndarray) -> float:
    n = len(resistance_matrix)
    growth = 0
    for i in range(n):
        B = {}
        for j in range(n):
            d = int(math.ceil(resistance_matrix[i][j]))
            if d not in B:
                B[d] = 1
            else:
                B[d] = B[d] + 1
        for d in range(n):
            if d in B and (d + 1) in B:
                ratio = B[d+1] / B[d]
                growth = max(growth, ratio)
    return growth


def er_density(resistances: list[float]) -> float:
    buckets = []

    m = len(resistances)
    n = (1 + math.sqrt(1 + 8*m)) / 2

    # One vertex in every 0-ball
    buckets.append(n)
    radius = 1
    count = 0
    resistances.sort()
    for r in resistances:
        if r <= radius:
            count += 1
        else:
            buckets.append(count)
            radius += 1
            count = 0
    ratios = []
    for i in range(len(buckets) - 1):
        ratios.append((buckets[i+1] / buckets[i]))
    return max(ratios)

if __name__ == '__main__':
    graphs = all_graph_dirs()
    for g_dir in graphs:
        for root, dirs, files in os.walk(g_dir):
            if F_RES_MAT not in files:
                print(f'No resistance matrix found for graph [{g_dir}]')

                # TODO: compute multiset from adjacency

                if F_ADJ_MAT not in files:

                    # No graph data, cannot analyze

                    print(f'No adjacency matrix found for graph[{g_dir}]. Ignoring graph [{g_dir}]')
            else:
                if F_RES_GRO in files:
                    print(f'ER-growth for graph [{g_dir}] has already been computed.')
                else:
                    print(f'Computing ER-growth for graph [{g_dir}]...')
                    resistance_matrix = []
                    with open(os.path.join(root, F_RES_MLT), 'r') as f:
                        resistance_matrix = np.array(json.load(f))
                    print(resistance_matrix)
                    density = er_growth(resistance_matrix)
                    with open(os.path.join(root, F_RES_GRO), 'w') as f:
                        f.write(json.dumps(density))
