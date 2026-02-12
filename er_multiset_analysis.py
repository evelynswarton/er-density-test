import yaml
import networkx as nx
import numpy as np
import os
from matplotlib import pyplot as plt
from typing import List, Tuple, Dict, Any


def read_resistances(path: str = "./results") -> List[Tuple[int, List[float]]]:
    """
    Read YAML files and extract (n, resistance_multiset) data.
    读取YAML文件并提取(n, resistance_multiset)数据。

    Args:
        path: Directory path containing YAML files / 包含YAML文件的目录路径

    Returns:
        List of tuples containing (n_nodes, resistance_multiset)
        包含(n_nodes, resistance_multiset)的元组列表
    """
    data = []

    if not os.path.exists(path):
        print(f"Warning: Directory {path} does not exist")
        return data

    yaml_files = [f for f in os.listdir(path) if f.endswith(".yaml")]

    for filename in yaml_files:
        filepath = os.path.join(path, filename)
        try:
            with open(filepath, "r") as file:
                graph_data = yaml.safe_load(file)

            n = graph_data["metadata"]["n_nodes"]
            resistance_multiset = graph_data.get("resistance_multiset", [])

            # Convert numpy scalars to regular floats
            resistance_multiset = convert_numpy_scalars(resistance_multiset)

            data.append((n, resistance_multiset))

        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    return data


def convert_numpy_scalars(multiset: List[Any]) -> List[float]:
    """
    Convert numpy scalar objects to regular Python floats.
    将numpy标量对象转换为常规Python浮点数。

    Args:
        multiset: List containing numpy scalars or numeric types
        包含numpy标量或数值类型的列表

    Returns:
        List of Python floats / Python浮点数列表
    """
    converted = []
    for item in multiset:
        if hasattr(item, "item"):  # numpy scalar
            converted.append(float(item.item()))
        elif isinstance(item, (int, float)):
            converted.append(float(item))
        else:
            try:
                converted.append(float(item))
            except (ValueError, TypeError):
                print(f"Warning: Could not convert {item} to float")
    return converted


def compute_density(n: int, er_multiset: List[float]) -> float:
    """
    Compute effective resistance density using histogram-based approach.
    使用基于直方图的方法计算有效电阻密度。

    Density is defined as the maximum frequency in any resistance bin.
    密度定义为任何电阻区间中的最大频率。

    Args:
        n: Number of nodes in the graph / 图中的节点数
        er_multiset: List of effective resistance values / 有效电阻值列表

    Returns:
        Density value representing the concentration of resistance values
        表示电阻值集中度的密度值
    """
    if not er_multiset:
        return 0.0

    # Sort resistance values for analysis
    sorted_multiset = sorted(er_multiset)

    # TODO: get density

    return density


def analyze(path: str = "./results") -> None:
    """
    Analyze effective resistance density across graph sizes.
    分析不同图规模下的有效电阻密度。

    Creates a scatter plot showing how density varies with graph size.
    创建散点图显示密度如何随图规模变化。

    Args:
        path: Directory containing graph data files / 包含图数据文件的目录
    """
    data = read_resistances(path)

    if not data:
        print(f"No data found in {path}")
        return

    x = []  # graph sizes
    y = []  # densities

    for graph in data:
        n = graph[0]
        er_multiset = graph[1]
        density = compute_density(n, er_multiset)
        x.append(n)
        y.append(density)
        print(f"Graph with {n} nodes: density = {density}")

    # Create scatter plot if data exists
    if x and y:
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, alpha=0.7, s=50)
        plt.xlabel("Number of Nodes (n)")
        plt.ylabel("Effective Resistance Density")
        plt.title("Effective Resistance Density vs Graph Size")
        plt.grid(True, alpha=0.3)

        # Sort by x for better visualization
        sorted_indices = np.argsort(x)
        x_sorted = np.array(x)[sorted_indices]
        y_sorted = np.array(y)[sorted_indices]

        plt.plot(x_sorted, y_sorted, "r-", alpha=0.3, label="Trend")
        plt.legend()

        plt.tight_layout()
        plt.savefig("density_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Analysis complete. Plot saved as density_analysis.png")
        print(f"Analyzed {len(x)} graphs")
        print(f"Density range: [{min(y):.4f}, {max(y):.4f}]")
    else:
        print("No data to plot")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze effective resistance density")
    parser.add_argument(
        "--path", default="./results", help="Directory containing graph data"
    )

    args = parser.parse_args()
    analyze(args.path)
