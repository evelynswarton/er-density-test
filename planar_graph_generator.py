#!/usr/bin/env python3
"""
Planar Graph Generator and Effective Resistance Calculator
平面图生成器与有效电阻计算器

This module generates planar graphs and computes the multiset of pairwise
effective resistance distances between all vertices.
本模块生成平面图并计算所有顶点对之间的有效电阻距离的多重集。
"""

import numpy as np
import networkx as nx
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import argparse
import json
import yaml
from datetime import datetime
import sys
import os

# Try to import Boltzmann sampler / 尝试导入玻尔兹曼采样器
try:
    # Add the boltzmann-planar-graph directory to Python path if needed
    # 如果需要，将boltzmann-planar-graph目录添加到Python路径
    boltzmann_path = Path(__file__).parent / "boltzmann-planar-graph"
    if boltzmann_path.exists():
        sys.path.insert(0, str(boltzmann_path))

    from planar_graph_sampler.planar_graph_generator import PlanarGraphGenerator

    BOLTZMANN_AVAILABLE = True
except ImportError as e:
    BOLTZMANN_AVAILABLE = False
    BOLTZMANN_IMPORT_ERROR = str(e)


def generate_planar_graph(
    n_vertices: int,
    seed: int,
    weighted: bool = False,
    method: str = "delaunay",
    epsilon: float = 0.1,
    require_connected: bool = True,
    with_embedding: bool = False,
    allow_multiproc: bool = False,
) -> nx.Graph:
    """
    Generate a random planar graph with specified number of vertices.
    生成具有指定顶点数的随机平面图。

    Args:
        n_vertices: Number of vertices in the graph / 图中顶点数
        seed: Random seed for reproducibility / 可重现性的随机种子
        weighted: Whether to assign random weights to edges / 是否为边分配随机权重
        method: Generation method ("delaunay" or "boltzmann") / 生成方法
        epsilon: Size tolerance for Boltzmann sampler (0=exact, >0=approximate) / 玻尔兹曼采样器的大小容差
        require_connected: Whether to require connectivity (Boltzmann only) / 是否要求连通性
        with_embedding: Whether to return planar embedding (Boltzmann only) / 是否返回平面嵌入
        allow_multiproc: Whether to allow parallel processing (Boltzmann only) / 是否允许并行处理

    Returns:
        NetworkX Graph object representing a planar graph / 表示平面图的NetworkX图对象

    Methods:
        - "delaunay": Uses Delaunay triangulation (default, maintains backward compatibility)
        - "boltzmann": Uses Boltzmann sampler for uniform random planar graphs
    生成方法：
        - "delaunay": 使用德劳内三角剖分（默认，保持向后兼容）
        - "boltzmann": 使用玻尔兹曼采样器生成均匀随机平面图

    Boltzmann advantages / 玻尔兹曼优势:
        - Uniform random sampling from all planar graphs / 从所有平面图均匀随机采样
        - Better size control for large graphs / 对大图更好的大小控制
        - Exact connectedness guarantee / 精确的连通性保证
        - Theoretical soundness / 理论上的正确性

    Delaunay advantages / 德劳内优势:
        - Faster for small graphs (< 100 nodes) / 对小图更快（<100节点）
        - Deterministic with same seed / 相同种子的确定性
        - No additional dependencies / 无额外依赖
    """
    if method == "delaunay":
        return _generate_planar_graph_delaunay(n_vertices, seed, weighted)
    elif method == "boltzmann":
        return _generate_planar_graph_boltzmann(
            n_vertices,
            seed,
            weighted,
            epsilon,
            require_connected,
            with_embedding,
            allow_multiproc,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'delaunay' or 'boltzmann'")


def _generate_planar_graph_delaunay(
    n_vertices: int, seed: int, weighted: bool = False
) -> nx.Graph:
    """Generate planar graph using Delaunay triangulation method.
    使用德劳内三角剖分方法生成平面图。"""
    rng = np.random.default_rng(seed)

    # Generate a random planar graph using Delaunay triangulation approach
    # 使用德劳内三角剖分方法生成随机平面图
    # Create random points in 2D and compute Delaunay triangulation
    # 在2D中创建随机点并计算德劳内三角剖分
    points = rng.random((n_vertices, 2))

    # Create Delaunay triangulation (guaranteed planar)
    # 创建德劳内三角剖分（保证平面性）
    from scipy.spatial import Delaunay

    tri = Delaunay(points)

    # Build graph from triangulation
    # 从三角剖分构建图
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))

    # Add edges from triangulation
    # 从三角剖分添加边
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                G.add_edge(simplex[i], simplex[j])

    # Randomly remove edges while maintaining connectivity
    # 随机移除边同时保持连通性
    edges = list(G.edges())
    rng.shuffle(edges)

    for edge in edges:
        if G.degree(edge[0]) > 2 and G.degree(edge[1]) > 2:
            G.remove_edge(*edge)
            if not nx.is_connected(G):
                G.add_edge(*edge)

    # Add weights if requested
    # 如果请求则添加权重
    if weighted:
        G = add_edge_weights(G, seed)

    return G


def _generate_planar_graph_boltzmann(
    n_vertices: int,
    seed: int,
    weighted: bool = False,
    epsilon: float = 0.1,
    require_connected: bool = True,
    with_embedding: bool = False,
    allow_multiproc: bool = False,
) -> nx.Graph:
    """Generate planar graph using Boltzmann sampler method.
    使用玻尔兹曼采样器方法生成平面图。"""
    if not BOLTZMANN_AVAILABLE:
        raise RuntimeError(
            f"Boltzmann sampler not available. Install with: pip install -e boltzmann-planar-graph\n"
            f"Import error: {BOLTZMANN_IMPORT_ERROR}"
        )

    if n_vertices < 3:
        raise ValueError("Boltzmann sampler requires at least 3 vertices")

    if not (0 <= epsilon <= 1):
        raise ValueError("epsilon must be between 0 and 1")

    # Set up random seeds for Boltzmann sampler
    import random

    random.seed(seed)
    np.random.seed(seed)

    try:
        # Create Boltzmann generator
        generator = PlanarGraphGenerator(
            n=n_vertices,
            epsilon=epsilon,
            require_connected=require_connected,
            with_embedding=with_embedding,
            allow_multiproc=allow_multiproc,
        )

        # Generate the graph
        G = generator.sample()

        # Convert to 0-indexed nodes (Boltzmann uses 1-indexed)
        if G.number_of_nodes() > 0:
            mapping = {node: node - 1 for node in G.nodes()}
            G = nx.relabel_nodes(G, mapping)

        # Verify planarity and connectivity requirements
        is_planar, _ = nx.check_planarity(G)
        if not is_planar:
            raise RuntimeError("Generated graph is not planar - this should not happen")

        if require_connected and not nx.is_connected(G):
            raise RuntimeError(
                "Generated graph is not connected - this should not happen"
            )

        print(
            f"Boltzmann generated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )

    except Exception as e:
        raise RuntimeError(f"Boltzmann sampler failed: {str(e)}")

    # Add weights if requested
    if weighted:
        G = add_edge_weights(G, seed + 1000)  # Use different seed for weights

    return G


def add_edge_weights(G: nx.Graph, seed: int) -> nx.Graph:
    """
    Add random weights to edges of a graph.
    为图的边添加随机权重。

    Args:
        G: NetworkX Graph object / NetworkX图对象
        seed: Random seed for reproducibility / 可重现性的随机种子

    Returns:
        NetworkX Graph object with weighted edges / 带有权重边的NetworkX图对象
    """
    rng = np.random.default_rng(seed)

    # Assign random weights using uniform distribution [0, 1]
    for edge in G.edges():
        weight = rng.random()  # Uniform [0, 1)
        G[edge[0]][edge[1]]["weight"] = weight

    return G


def adjacency_matrix_to_numpy(G: nx.Graph) -> np.ndarray:
    """
    Convert NetworkX graph to numpy adjacency matrix.

    Args:
        G: NetworkX Graph object

    Returns:
        Numpy adjacency matrix
    """
    return nx.to_numpy_array(G, dtype=np.float64)


def compute_effective_resistance(G: nx.Graph) -> np.ndarray:
    """
    Compute effective resistance distances between all pairs of vertices.
    计算所有顶点对之间的有效电阻距离。

    Uses the Moore-Penrose pseudoinverse of the Laplacian matrix.
    使用拉普拉斯矩阵的摩尔-彭罗斯伪逆。
    R_ij = L^+_ii + L^+_jj - 2*L^+_ij

    Args:
        G: NetworkX Graph object / NetworkX图对象

    Returns:
        Symmetric matrix of effective resistance distances / 有效电阻距离的对称矩阵
    """
    n = G.number_of_nodes()

    # Compute Laplacian matrix
    L = nx.laplacian_matrix(G).astype(np.float64).toarray()

    # Compute Moore-Penrose pseudoinverse
    # Use eigenvalue decomposition for numerical stability
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Remove zero eigenvalue (first one for connected graph)
    eigenvalues = eigenvalues[1:]
    eigenvectors = eigenvectors[:, 1:]

    # Compute pseudoinverse
    L_pinv = eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T

    # Compute effective resistance matrix
    resistance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            resistance_matrix[i, j] = L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j]

    return resistance_matrix


def get_resistance_multiset(resistance_matrix: np.ndarray) -> List[float]:
    """
    Extract multiset of resistance distances from upper triangular matrix.
    提取上三角矩阵中的电阻距离多重集。

    Args:
        resistance_matrix: Symmetric matrix of effective resistance distances
        有效电阻距离的对称矩阵

    Returns:
        List of all pairwise resistance distances / 所有成对电阻距离的列表
    """
    n = resistance_matrix.shape[0]
    multiset = []

    for i in range(n):
        for j in range(i + 1, n):
            multiset.append(float(resistance_matrix[i, j]))

    return multiset


def generate_filename(n_nodes: int, m_edges: int, weighted: bool, graph_id: int) -> str:
    """
    Generate descriptive filename for graph data.
    生成描述性的图数据文件名。

    Args:
        n_nodes: Number of nodes in the graph / 图中节点数
        m_edges: Number of edges in the graph / 图中边数
        weighted: Whether the graph is weighted / 图是否加权
        graph_id: Identifier for the graph / 图标识符

    Returns:
        Descriptive filename string / 描述性文件名字符串
    """
    weight_status = "weighted" if weighted else "unweighted"
    return f"graph_n{n_nodes}_m{m_edges}_{weight_status}_id{graph_id}.yaml"


def save_results(
    adjacency_matrix: np.ndarray,
    resistance_multiset: List[float],
    graph_metadata: dict,
    output_dir: Path,
    graph_id: Optional[int] = None,
) -> None:
    """
    Save adjacency matrix, resistance multiset, and metadata to single YAML file.
    将邻接矩阵、电阻多重集和元数据保存到单个YAML文件。

    Args:
        adjacency_matrix: Numpy adjacency matrix / NumPy邻接矩阵
        resistance_multiset: List of resistance distances / 电阻距离列表
        graph_metadata: Dictionary containing graph metadata / 包含图元数据的字典
        output_dir: Directory to save results / 保存结果的目录
        graph_id: Optional graph identifier / 可选的图标识符
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate descriptive filename
    actual_graph_id = graph_id if graph_id is not None else graph_metadata["graph_id"]
    filename = generate_filename(
        graph_metadata["n_nodes"],
        graph_metadata["m_edges"],
        graph_metadata["weighted"],
        actual_graph_id,
    )

    filepath = output_dir / filename

    # Create comprehensive data structure
    graph_data = {
        "metadata": graph_metadata,
        "adjacency_matrix": adjacency_matrix.tolist(),
        "resistance_multiset": resistance_multiset,
    }

    # Save all data in single YAML file
    with open(filepath, "w") as f:
        yaml.dump(graph_data, f, default_flow_style=False)

    print(f"Saved results to {output_dir}")
    print(f"  Combined file: {filepath}")
    print(f"  Format: YAML")
    print(f"  Contains: adjacency matrix, resistance multiset, metadata")


def load_results(filepath: Path) -> dict:
    """
    Load graph data from YAML file.
    从YAML文件加载图数据。

    Args:
        filepath: Path to the YAML file / YAML文件路径

    Returns:
        Dictionary containing adjacency matrix, resistance multiset, and metadata
        包含邻接矩阵、电阻多重集和元数据的字典
    """
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)

    return {
        "adjacency_matrix": np.array(data["adjacency_matrix"]),
        "resistance_multiset": data["resistance_multiset"],
        "metadata": data["metadata"],
    }


def main() -> None:
    """Main function to generate planar graphs and compute effective resistances.
    生成平面图并计算有效电阻的主函数。"""
    parser = argparse.ArgumentParser(
        description="Generate planar graphs and compute effective resistance distances"
    )
    parser.add_argument(
        "--n-vertices",
        type=int,
        default=10,
        help="Number of vertices in the planar graph",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--num-graphs", type=int, default=1, help="Number of graphs to generate"
    )
    parser.add_argument(
        "--weighted", action="store_true", help="Generate weighted graphs"
    )
    parser.add_argument(
        "--method",
        choices=["delaunay", "boltzmann"],
        default="delaunay",
        help="Graph generation method (default: delaunay)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Size tolerance for Boltzmann sampler (0=exact, >0=approximate, default: 0.1)",
    )
    parser.add_argument(
        "--require-connected",
        action="store_true",
        default=True,
        help="Require connectivity for Boltzmann sampler (default: True)",
    )
    parser.add_argument(
        "--allow-multiproc",
        action="store_true",
        help="Allow parallel processing for Boltzmann sampler",
    )
    parser.add_argument(
        "--compare-methods",
        action="store_true",
        help="Generate graphs using both methods and compare them",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Check if Boltzmann is available
    if args.method == "boltzmann" and not BOLTZMANN_AVAILABLE:
        print(
            "Warning: Boltzmann sampler not available. Falling back to Delaunay method."
        )
        print(f"Import error: {BOLTZMANN_IMPORT_ERROR}")
        args.method = "delaunay"

    # Adjust seed for Boltzmann method compatibility
    if args.method == "boltzmann":
        print(f"\nUsing Boltzmann sampler with parameters:")
        print(f"  Vertices: {args.n_vertices}")
        print(f"  Epsilon: {args.epsilon} (size tolerance)")
        print(f"  Require connected: {args.require_connected}")
        print(f"  Allow multiproc: {args.allow_multiproc}")
        if not BOLTZMANN_AVAILABLE:
            print(f"  Warning: Boltzmann not available, using Delaunay instead")

    for graph_id in range(args.num_graphs):
        method_to_use = args.method

        if args.compare_methods:
            # Generate both methods
            methods_to_test = ["delaunay"]
            if BOLTZMANN_AVAILABLE:
                methods_to_test.append("boltzmann")
        else:
            methods_to_test = [method_to_use]

        for method in methods_to_test:
            print(
                f"\nGenerating graph {graph_id + 1}/{args.num_graphs} (method: {method})"
            )
            print(
                f"Vertices: {args.n_vertices}, Seed: {args.seed + graph_id}, Weighted: {args.weighted}"
            )

            # Generate planar graph
            try:
                G = generate_planar_graph(
                    args.n_vertices,
                    args.seed + graph_id,
                    args.weighted,
                    method=method,
                    epsilon=args.epsilon,
                    require_connected=args.require_connected,
                    allow_multiproc=args.allow_multiproc,
                )
            except Exception as e:
                print(f"Error generating graph with {method}: {e}")
                continue

        # Convert to adjacency matrix
        adj_matrix = adjacency_matrix_to_numpy(G)

        # Compute effective resistance
        resistance_matrix = compute_effective_resistance(G)
        resistance_multiset = get_resistance_multiset(resistance_matrix)

        # Create metadata dictionary
        graph_metadata = {
            "n_nodes": G.number_of_nodes(),
            "m_edges": G.number_of_edges(),
            "weighted": args.weighted,
            "graph_id": graph_id,
            "seed": args.seed + graph_id,
            "is_connected": nx.is_connected(G),
            "is_planar": nx.check_planarity(G)[0],
            "node_degrees": [d for n, d in G.degree()],
            "edge_weights": [
                G[edge[0]][edge[1]].get("weight", 1.0) for edge in G.edges()
            ]
            if args.weighted
            else None,
            "generation_method": method,
            "boltzmann_epsilon": args.epsilon if method == "boltzmann" else None,
            "boltzmann_connected": args.require_connected
            if method == "boltzmann"
            else None,
        }

        # Modify graph_id if comparing methods
        current_graph_id = graph_id
        if args.compare_methods:
            if method == "delaunay":
                current_graph_id = graph_id * 2
            else:
                current_graph_id = graph_id * 2 + 1

        # Save results
        save_results(
            adj_matrix,
            resistance_multiset,
            graph_metadata,
            output_dir,
            current_graph_id,
        )

        # Print summary statistics
        print(f"  Graph edges: {G.number_of_edges()}")
        print(f"  Resistance distances: {len(resistance_multiset)} pairs")
        print(f"  Mean resistance: {np.mean(resistance_multiset):.4f}")
        print(f"  Std resistance: {np.std(resistance_multiset):.4f}")
        if args.weighted:
            weights = [G[edge[0]][edge[1]]["weight"] for edge in G.edges()]
            print(f"  Weight range: [{min(weights):.3f}, {max(weights):.3f}]")


if __name__ == "__main__":
    main()
