# Planar Graph Generation and Effective Resistance Analysis

A research codebase for generating planar graphs and computing effective resistance distances between all vertex pairs. Implements both Delaunay triangulation and Boltzmann sampling methods for comparative analysis.

## Quick Start / 快速开始

```bash
# Install dependencies / 安装依赖
pip install -r requirements.txt

# Generate a single planar graph with effective resistance (default: Delaunay) / 生成单个平面图及其有效电阻（默认：德劳内三角剖分）
python planar_graph_generator.py --n-vertices 10 --seed 42

# Generate weighted graph using Boltzmann method / 使用玻尔兹曼方法生成加权图
python planar_graph_generator.py --n-vertices 15 --method boltzmann --weighted

# Generate multiple graphs for analysis
python planar_graph_generator.py --n-vertices 25 --num-graphs 5 --seed 123
```

## Overview

This codebase provides:
- **Planar graph generation** using two methods:
  - Delaunay triangulation (default, always available)
  - Boltzmann sampling (uniform random planar graphs)
- **Effective resistance computation** for all vertex pairs

## Installation / 安装指南

```bash
# Clone repository / 克隆仓库
git clone github.com/evelynswarton/er-density-test
cd er-density-test

# Create virtual environment (recommended) / 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate / Windows系统请使用：venv\Scripts\activate

# Install dependencies / 安装依赖
pip install -r requirements.txt
```

**Note / 注意**: The Boltzmann sampler is optional. If unavailable, the code automatically falls back to Delaunay triangulation. / 玻尔兹曼采样器是可选的。如果不可用，代码将自动回退到德劳内三角剖分。

## Usage / 使用方法

### Generate Planar Graphs

```bash
# Basic generation (Delaunay, unweighted)
python planar_graph_generator.py --n-vertices 10 --seed 42

# Weighted graph generation
python planar_graph_generator.py --n-vertices 10 --weighted --seed 42

# Boltzmann sampling (if available)
python planar_graph_generator.py --n-vertices 10 --method boltzmann --seed 42

# Multiple graphs
python planar_graph_generator.py --n-vertices 10 --num-graphs 5 --seed 42

# Custom output directory
python planar_graph_generator.py --n-vertices 10 --output-dir my_results
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n-vertices` | Number of vertices in graph | 10 |
| `--seed` | Random seed for reproducibility | 42 |
| `--method` | Generation method (`delaunay`/`boltzmann`) | `delaunay` |
| `--weighted` | Generate weighted graphs | False |
| `--num-graphs` | Number of graphs to generate | 1 |
| `--epsilon` | Delaunay noise parameter | 0.1 |
| `--output-dir` | Results directory | `results` |

## Output Files

Generated files are saved in the specified output directory (default: `results/`):

- `graph_n{N}_m{M}_{weighted/unweighted}_id{ID}.yaml` - Complete graph data including:
  - Adjacency matrix
  - Vertex positions (Delaunay method)
  - Effective resistance multiset
  - Graph metadata

- `graph_n{N}_m{M}_{weighted/unweighted}_id{ID}.npz` - Compressed adjacency matrix

### YAML Structure

```yaml
# Example generated YAML file
metadata:
  n_vertices: 10
  n_edges: 15
  method: delaunay
  weighted: false
  seed: 42
adjacency_matrix:
  - [0.0, 1.0, 0.0, ...]  # Row 0
  - [1.0, 0.0, 1.0, ...]  # Row 1
  ...
effective_resistance:
  pairwise_distances: [0.5, 1.2, 0.8, ...]
  multiset: [0.5, 0.5, 0.8, 1.2, ...]
```

## Reproducibility

To reproduce specific results:

```bash
# Exact graph reproduction
python planar_graph_generator.py --n-vertices 10 --seed 42 --method delaunay

# Comparison reproduction
python compare_methods.py --test-size 20 --num-trials 5 --seed 42
```

**Critical parameters for reproducibility:**
- Always specify `--seed`
- Note the generation method (`--method`)
- Record graph size and weighted/unweighted status
- Use same Python environment (requirements.txt)

## Troubleshooting

### Boltzmann Sampler Issues
```bash
# Check if Boltzmann sampler is available
python -c "from planar_graph_generator import BOLTZMANN_AVAILABLE; print('Boltzmann:', BOLTZMANN_AVAILABLE)"
```

If unavailable, ensure `boltzmann-planar-graph/` directory is present and dependencies installed.

### Memory Issues
For large graphs (>100 vertices), consider:
- Reducing `--num-graphs`
- Using `--num-trials 1` for comparisons
- Monitoring memory usage during benchmarking

### Import Errors
```bash
# Verify installation
python -c "import networkx, numpy, scipy, yaml; print('All dependencies OK')"
```

## File Structure

```
├── planar_graph_generator.py    # Main generation script
├── requirements.txt             # Python dependencies
├── boltzmann-planar-graph/      # Boltzmann sampler implementation
└── results/                     # Generated graphs and analysis
```

## References

Inherited from boltzmann sampling submodule
- Fusy, É. (2005). "Quadratic exact-size and linear approximate-size random generation of planar graphs."
- Fusy, É. (2009). "Uniform random sampling of planar graphs in linear time." Random Structures & Algorithms 35.4: 464-522.
