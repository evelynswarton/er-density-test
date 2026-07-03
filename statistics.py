import os
import re
import json
import numpy as np
from collections import defaultdict

GRAPHS_DIR = './graphs/planar'
METRIC_GROWTH_FNAME = 'metric_growth.json'
TARGET_SIZES = [8, 64, 512, 2048]
EPSILON = 0.1


def all_graph_dirs():
    graph_dirs = []
    for method in ('boltzmann', 'delaunay'):
        method_dir = os.path.join(GRAPHS_DIR, method)
        if not os.path.isdir(method_dir):
            continue
        for entry in sorted(os.listdir(method_dir)):
            entry_path = os.path.join(method_dir, entry)
            if os.path.isdir(entry_path):
                graph_dirs.append(entry_path)
    return graph_dirs


def parse_graph_name(basename):
    m = re.match(r'(\d+)verts_(\d+)edges_([0-9a-f]+)', basename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3)


def assign_target(n_verts, targets, eps):
    for target in targets:
        if abs(n_verts - target) / target <= eps:
            return target
    return None


def parse_metric_growth(filepath):
    with open(filepath) as f:
        text = f.read()

    result = {}
    for m in re.finditer(r'\{\{(.*?)\}\}', text, re.DOTALL):
        snippet = text[max(0, m.start() - 10): m.start()]
        if 'AVG' in snippet:
            key = 'AVG'
        elif 'MAX' in snippet:
            key = 'MAX'
        else:
            continue
        try:
            result[key] = json.loads('{' + m.group(1) + '}')
        except json.JSONDecodeError:
            pass
    return result


def get_method(dirpath):
    return os.path.relpath(dirpath, GRAPHS_DIR).split(os.sep)[0]


def collect_data():
    records = []
    for gdir in all_graph_dirs():
        method = get_method(gdir)
        parsed = parse_graph_name(os.path.basename(gdir))
        if parsed is None:
            continue
        n_verts, n_edges, _ = parsed
        target = assign_target(n_verts, TARGET_SIZES, EPSILON)
        if target is None:
            continue

        mg_path = os.path.join(gdir, METRIC_GROWTH_FNAME)
        if not os.path.isfile(mg_path):
            continue

        mg_data = parse_metric_growth(mg_path)
        for stat_type in ('AVG', 'MAX'):
            if stat_type not in mg_data:
                continue
            for dist_fn, value in mg_data[stat_type].items():
                records.append({
                    'method': method,
                    'target': target,
                    'stat_type': stat_type,
                    'dist_fn': dist_fn,
                    'value': value,
                })

    return records


def aggregate(records):
    groups = defaultdict(list)
    for rec in records:
        key = (rec['method'], rec['target'], rec['stat_type'], rec['dist_fn'])
        groups[key].append(rec['value'])

    agg = {}
    for key, values in groups.items():
        arr = np.array(values)
        agg[key] = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            'count': len(arr),
        }
    return agg


DIST_FN_LABELS = {
    'shortest path': 'Shortest Path',
    'resistance distance': 'Resistance Distance'
}


def cell(mean, std):
    if std < 1e-10:
        return f'${mean:.2f}$'
    return f'${mean:.2f}\\pm{std:.2f}$'


def print_latex(agg, method, dist_fns, targets):
    k = len(dist_fns)
    col_spec = 'c|' + 'cc' * k

    print(f'% --- {method} ---')
    print(r'\begin{tabular}{' + col_spec + '}')
    print('  $n$ & ' + ' & '.join(
        r'\multicolumn{2}{c}{' + DIST_FN_LABELS.get(fn, fn) + '}' for fn in dist_fns
    ) + r' \\')
    print('  & ' + ' & '.join(
        r'$\bar{x}$ & $\max$' for _ in dist_fns
    ) + r' \\')
    print(r'  \hline')

    for target in targets:
        cells = [f'  {target}']
        ok = False
        for fn in dist_fns:
            avg_key = (method, target, 'AVG', fn)
            max_key = (method, target, 'MAX', fn)
            if avg_key in agg and max_key in agg:
                cells.append(cell(agg[avg_key]['mean'], agg[avg_key]['std']))
                cells.append(cell(agg[max_key]['mean'], agg[max_key]['std']))
                ok = True
            else:
                cells.append('---')
                cells.append('---')
        if ok:
            print(' & '.join(cells) + r' \\')

    print(r'\end{tabular}')
    print()


def main():
    records = collect_data()
    agg = aggregate(records)

    dist_fns = [
        'shortest path',
        'resistance distance',
    ]
    targets = [8, 64, 512, 2048]

    for method in ('boltzmann', 'delaunay'):
        print_latex(agg, method, dist_fns, targets)


if __name__ == '__main__':
    main()
