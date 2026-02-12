# Planar Graph Generation and Effective Resistance Analysis

A research codebase for generating planar graphs and computing effective resistance distances between all vertex pairs. Implements both Delaunay triangulation and Boltzmann sampling methods for comparative analysis.

本研究代码库用于生成平面图并计算所有顶点对之间的有效电阻距离。实现了德劳内三角剖分和玻尔兹曼采样两种方法进行对比分析。

## Quick Start / 快速开始

```bash
# Install dependencies / 安装依赖
pip install -r requirements.txt

# Generate a single planar graph with effective resistance (default: Delaunay) / 生成单个平面图及其有效电阻（默认：德劳内三角剖分）
python planar_graph_generator.py --n-vertices 10 --seed 42

# Generate weighted graph using Boltzmann method / 使用玻尔兹曼方法生成加权图
python planar_graph_generator.py --n-vertices 15 --method boltzmann --weighted

# Generate multiple graphs for analysis / 生成多个图用于分析
python planar_graph_generator.py --n-vertices 25 --num-graphs 5 --seed 123
```

## Overview / 概述

This codebase provides:
- **Planar graph generation** using two methods:
  - Delaunay triangulation (default, always available) / 德劳内三角剖分（默认，始终可用）
  - Boltzmann sampling (uniform random planar graphs) / 玻尔兹曼采样（均匀随机平面图）
- **Effective resistance computation** for all vertex pairs / 所有顶点对的有效电阻计算

本研究代码库提供：
- **平面图生成**：使用两种方法
  - 德劳内三角剖分（默认方法，始终可用）
  - 玻尔兹曼采样（均匀随机平面图）
- **有效电阻计算**：计算所有顶点对之间的有效电阻距离

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

### Generate Planar Graphs / 生成平面图

```bash
# Basic generation (Delaunay, unweighted) / 基础生成（德劳内，无权重）
python planar_graph_generator.py --n-vertices 10 --seed 42

# Weighted graph generation / 加权图生成
python planar_graph_generator.py --n-vertices 10 --weighted --seed 42

# Boltzmann sampling (if available) / 玻尔兹曼采样（如果可用）
python planar_graph_generator.py --n-vertices 10 --method boltzmann --seed 42

# Multiple graphs / 多个图
python planar_graph_generator.py --n-vertices 10 --num-graphs 5 --seed 42

# Custom output directory / 自定义输出目录
python planar_graph_generator.py --n-vertices 10 --output-dir my_results
```

### Key Parameters / 关键参数

| Parameter | Description | Default | 描述 |
|-----------|-------------|---------|---------|
| `--n-vertices` | Number of vertices in graph | 10 | 图中顶点数量 |
| `--seed` | Random seed for reproducibility | 42 | 可重现性的随机种子 |
| `--method` | Generation method (`delaunay`/`boltzmann`) | `delaunay` | 生成方法 |
| `--weighted` | Generate weighted graphs | False | 生成加权图 |
| `--num-graphs` | Number of graphs to generate | 1 | 要生成的图数量 |
| `--epsilon` | Delaunay noise parameter | 0.1 | 德劳内噪声参数 |
| `--output-dir` | Results directory | `results` | 结果目录 |

## Output Files / 输出文件

Generated files are saved in the specified output directory (default: `results/`):
生成的文件保存在指定的输出目录中（默认：`results/`）：

- `graph_n{N}_m{M}_{weighted/unweighted}_id{ID}.yaml` - Complete graph data including:
  - Adjacency matrix / 邻接矩阵
  - Vertex positions (Delaunay method) / 顶点位置（德劳内方法）
  - Effective resistance multiset / 有效电阻多重集
  - Graph metadata / 图元数据

- `graph_n{N}_m{M}_{weighted/unweighted}_id{ID}.npz` - Compressed adjacency matrix / 压缩邻接矩阵

### YAML Structure / YAML结构

```yaml
# Example generated YAML file / 示例生成的YAML文件
metadata:
  n_vertices: 10  # Number of vertices / 顶点数量
  n_edges: 15     # Number of edges / 边数量
  method: delaunay # Generation method / 生成方法
  weighted: false  # Whether graph is weighted / 图是否加权
  seed: 42        # Random seed used / 使用的随机种子
adjacency_matrix:
  - [0.0, 1.0, 0.0, ...]  # Row 0 / 第0行
  - [1.0, 0.0, 1.0, ...]  # Row 1 / 第1行
  ...
effective_resistance:
  pairwise_distances: [0.5, 1.2, 0.8, ...]  # Pairwise distances / 成对距离
  multiset: [0.5, 0.5, 0.8, 1.2, ...]        # Multiset of distances / 距离的多重集
```

## Reproducibility / 可重现性

To reproduce specific results:
要重现特定结果：

```bash
# Exact graph reproduction / 精确图重现
python planar_graph_generator.py --n-vertices 10 --seed 42 --method delaunay

# Comparison reproduction / 比较重现
python compare_methods.py --test-size 20 --num-trials 5 --seed 42
```

**Critical parameters for reproducibility: / 可重现性的关键参数：**
- Always specify `--seed` / 始终指定随机种子
- Note the generation method (`--method`) / 记录生成方法
- Record graph size and weighted/unweighted status / 记录图大小和加权/无权重状态
- Use same Python environment (requirements.txt) / 使用相同的Python环境

## Troubleshooting / 故障排除

### Boltzmann Sampler Issues / 玻尔兹曼采样器问题
```bash
# Check if Boltzmann sampler is available / 检查玻尔兹曼采样器是否可用
python -c "from planar_graph_generator import BOLTZMANN_AVAILABLE; print('Boltzmann:', BOLTZMANN_AVAILABLE)"
```

If unavailable, ensure `boltzmann-planar-graph/` directory is present and dependencies installed.
如果不可用，确保 `boltzmann-planar-graph/` 目录存在并安装了依赖项。

### Memory Issues / 内存问题
For large graphs (>100 vertices), consider:
对于大图（>100个顶点），考虑：
- Reducing `--num-graphs` / 减少 `--num-graphs`
- Using `--num-trials 1` for comparisons / 比较时使用 `--num-trials 1`
- Monitoring memory usage during benchmarking / 在基准测试期间监控内存使用量

### Import Errors / 导入错误
```bash
# Verify installation / 验证安装
python -c "import networkx, numpy, scipy, yaml; print('All dependencies OK')"
```

## File Structure / 文件结构

```
├── planar_graph_generator.py    # Main generation script / 主生成脚本
├── requirements.txt             # Python dependencies / Python依赖
├── boltzmann-planar-graph/      # Boltzmann sampler implementation / 玻尔兹曼采样器实现
└── results/                     # Generated graphs and analysis / 生成的图和分析
```

## References / 参考文献

Inherited from boltzmann sampling submodule / 继承自玻尔兹曼采样子模块
- Fusy
