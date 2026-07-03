# ER Density Testing - Heuristic Analyzer for Effective Resistance Volume Growth

A research codebase for generating planar graphs and computing effective resistance distances between all vertex pairs. Implements both Delaunay triangulation and Boltzmann sampling methods for comparative analysis.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate a single planar graph with effective resistance (default: Delaunay)
python planar_graph_generator.py --n-vertices 10 --seed 42

# Generate weighted graph using Boltzmann method
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

## Installation

```bash
# Clone repository
git clone github.com/evelynswarton/er-density-test
cd er-density-test

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

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
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n-vertices` | Number of vertices in graph | 10 |
| `--seed` | Random seed for reproducibility | 42 |
| `--method` | Generation method (`delaunay`/`boltzmann`) | `delaunay` |
| `--weighted` | Generate weighted graphs | False |
| `--max-degree` | Rejection sampling based on graph degree | Inf |
| `--num-graphs` | Number of graphs to generate | 1 |
| `--epsilon` | Size tolerance for Boltzmann sampler | 0.1 |

## Output Files

Generated files are saved in `./graphs/planar/{method}/{n}verts_{m}edges_{hash}/`:

- `adjacency_matrix.json` - Adjacency matrix
- `resistance_matrix.json` - Resistance matrix
- `resistance_multiset.json` - Resistance multiset

## Reproducibility

To reproduce specific results:

```bash
# Exact graph reproduction
python planar_graph_generator.py --n-vertices 10 --seed 42 --method delaunay
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
- Monitoring memory usage during benchmarking

### Import Errors
```bash
# Verify installation
python -c "import networkx, numpy, scipy, yaml; print('All dependencies OK')"
```

## File Structure

```
├── planar_graph_generator.py    # Main generation script
├── conversion.py                # File and matrix conversions
├── distances.py                 # Spectral distance measures and shortest path length
├── generate_all.py              # Batch generation script
├── growth_test.py               # Checking the growth rates of cached graphs
├── growth.py                    # Functions for well-defined effective resistance volume growths and the cumulative distribution function
├── statistics.py                # Gives a LaTeX table for results
├── requirements.txt             # Python dependencies
├── boltzmann-planar-graph/      # Boltzmann sampler implementation
└── graphs/                     # Generated graphs
```

## References

Inherited from boltzmann sampling submodule
- Fusy
