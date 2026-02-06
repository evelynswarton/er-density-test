# Planar Graph Generation and Effective Resistance Analysis

A research codebase for generating planar graphs and computing effective resistance distances between all vertex pairs. Implements both Delaunay triangulation and Boltzmann sampling methods for comparative analysis.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate a single planar graph with effective resistance (default: Delaunay)
python planar_graph_generator.py --n-vertices 10 --seed 42

# Generate weighted graph using Boltzmann method
python planar_graph_generator.py --n-vertices 15 --method boltzmann --weighted

# Compare both generation methods with benchmarking
python compare_methods.py --test-size 20 --num-trials 5

# Generate multiple graphs for analysis
python planar_graph_generator.py --n-vertices 25 --num-graphs 5 --seed 123
```

## Overview

This codebase provides:
- **Planar graph generation** using two methods:
  - Delaunay triangulation (default, always available)
  - Boltzmann sampling (uniform random planar graphs)
- **Effective resistance computation** for all vertex pairs
- **Comparative analysis** between generation methods
- **Benchmarking tools** for performance and statistical properties

## Installation

```bash
# Clone repository
git clone <repository-url>
cd some-random-code

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: The Boltzmann sampler is optional. If unavailable, the code automatically falls back to Delaunay triangulation.

## Usage

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

### Compare Generation Methods

```bash
# Full comparison with benchmarking
python compare_methods.py

# Custom comparison parameters
python compare_methods.py --test-size 25 --num-trials 10 --weighted

# Skip benchmarking, detailed comparison only
python compare_methods.py --skip-benchmark --test-size 15

# Custom benchmark sizes
python compare_methods.py --benchmark-sizes 10 20 40 80 --num-trials 3
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
- Comparison results in `comparison_results/` with benchmark statistics

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
├── planar_graph_generator.py     # Main generation script
├── compare_methods.py           # Method comparison and benchmarking
├── requirements.txt             # Python dependencies
├── boltzmann-planar-graph/      # Boltzmann sampler implementation
├── results/                     # Generated graphs and analysis
├── test_yaml_basic.ipynb       # Basic testing notebook
└── test_yaml_performance.ipynb # Performance testing notebook
```

## References

- Fusy, É. (2005). "Quadratic exact-size and linear approximate-size random generation of planar graphs."
- Fusy, É. (2009). "Uniform random sampling of planar graphs in linear time." Random Structures & Algorithms 35.4: 464-522.