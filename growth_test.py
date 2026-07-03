import distances as dst
import conversions
import os
import sys
import string
import json
import math
import numpy as np

graphs_dir = './graphs/'
adj_fname  = 'adjacency_matrix.json'
metric_growth_fname = 'metric_growth.json'

def dict_max_update(d, k, v):
    if k not in d:
        d[k] = v
    else:
        d[k] = max(d[k], v)

def metric_growth_stats(L):
    print('Computing distance matrices...')
    shortest_path = dst.__ALL_PAIRS_SHORTEST_PATH(L)
    L_PI = np.linalg.pinv(L)
    print('Computing all pairs resistance distance...')
    R = dst.__QUAD_FORM_MATRIX(L_PI)
    L_PI_SQ = np.square(L_PI)
    print('Computing all pairs biharmonic distance...')
    biharmonic_dist_mat = dst.__QUAD_FORM_MATRIX(L_PI_SQ)
    print('Computing all pairs triharmonic distance...')
    triharmonic_dist_mat = dst.__QUAD_FORM_MATRIX(np.matmul(L_PI, L_PI_SQ))
    print('Computing all pairs quadharmonic distance...')
    quadharmonic_dist_mat = dst.__QUAD_FORM_MATRIX(np.square(L_PI_SQ))
    growth_maxes = {}
    growth_avgs = {}
    growth_avgs['shortest path']=[]
    growth_avgs['resistance distance']=[]
    growth_avgs['biharmonic distance']=[]
    growth_avgs['triharmonic distance']=[]
    growth_avgs['quadharmonic distance']=[]
    n = len(L)
    for u in range(n):
        shortest_path_ball = {u}
        resistance_distance_ball = {u}
        biharmonic_distance_ball = {u}
        triharmonic_distance_ball = {u}
        quadharmonic_distance_ball = {u}
        unvisited = set(range(n))
        epsilon = 0
        while len(unvisited) > 0:
            shortest_path_size = len(shortest_path_ball)
            resistance_distance_size = len(resistance_distance_ball)
            biharmonic_distance_size = len(biharmonic_distance_ball)
            triharmonic_distance_size = len(triharmonic_distance_ball)
            quadharmonic_distance_size = len(quadharmonic_distance_ball)
            print(f'u={u}, eps={epsilon}, sizeS | sp[{shortest_path_size}], r[{resistance_distance_size}], bh[{biharmonic_distance_size}]')
            epsilon += 1
            for v in range(n):
                if shortest_path[u,v] <= epsilon:
                    shortest_path_ball.add(v)
                    unvisited.discard(v)
                if R[u,v] <= epsilon:
                    resistance_distance_ball.add(v)
                    unvisited.discard(v)
                if biharmonic_dist_mat[u,v] <= epsilon:
                    biharmonic_distance_ball.add(v)
                    unvisited.discard(v)
                if triharmonic_dist_mat[u,v] <= epsilon:
                    triharmonic_distance_ball.add(v)
                    unvisited.discard(v)
                if quadharmonic_dist_mat[u,v] <= epsilon:
                    quadharmonic_distance_ball.add(v)
                    unvisited.discard(v)
            ratio = len(shortest_path_ball) / shortest_path_size
            dict_max_update(growth_maxes, 'shortest path', ratio)
            growth_avgs['shortest path'].append(ratio)
            ratio = len(resistance_distance_ball) / resistance_distance_size
            dict_max_update(growth_maxes, 'resistance distance', ratio)
            growth_avgs['resistance distance'].append(ratio)
            ratio = len(biharmonic_distance_ball) / biharmonic_distance_size
            dict_max_update(growth_maxes, 'biharmonic distance', ratio)
            growth_avgs['biharmonic distance'].append(ratio)
            ratio = len(triharmonic_distance_ball) / triharmonic_distance_size
            dict_max_update(growth_maxes, 'triharmonic distance', ratio)
            growth_avgs['triharmonic distance'].append(ratio)
            ratio = len(quadharmonic_distance_ball) / quadharmonic_distance_size
            dict_max_update(growth_maxes, 'quadharmonic distance', ratio)
            growth_avgs['quadharmonic distance'].append(ratio)
    for k in growth_avgs:
        growth_avgs[k] = sum(growth_avgs[k]) / len(growth_avgs[k])
    return growth_maxes, growth_avgs

def all_graph_dirs() -> list[os.path]:
    graph_dirs = []
    for root, dirs, files in os.walk(graphs_dir):
        for name in dirs:
            if name not in ['graphs', 'planar', 'delaunay']:
                graph_dirs.append(os.path.join(root, name))
    return graph_dirs

if __name__ == '__main__':
    graphs = all_graph_dirs()
    for g_dir in graphs:
        for root, dirs, files in os.walk(g_dir):
            if adj_fname not in files:
                print(f'No adjacency matrix fount for G_DIR=[{g_dir}]')
            else:
                print(f'Computing metric growths for G=[{root.split('/')[-1]}]...')
                A = conversions.__MATRIX_FROM_FILE(os.path.join(root, adj_fname))
                L = conversions.__ADJACENCY_TO_LAPLACIAN(A)
                max_metric_growths, avg_metric_growths = metric_growth_stats(L)
                with open(os.path.join(root, metric_growth_fname), 'w') as FILE:
                    FILE.write('MAX: {' + json.dumps(max_metric_growths))
                    FILE.write('} \nAVG: {' + json.dumps(avg_metric_growths) + '}\n')
                    print(f'Saved to file [{root.split('/')[-1]}/{metric_growth_fname}].')
