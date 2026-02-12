import yaml
import networkx as nx
import numpy as np
import os
from matplotlib import pyplot as plt


def read_resistances(path: str = "./results"):
    """Read YAML files and extract (n, resistance_multiset) data."""
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


def convert_numpy_scalars(multiset):
    """Convert numpy scalar objects to regular Python floats."""
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


def compute_density(n, er_multiset):
    """Compute density: max(r_i) where r_i = count(≤ i+1) - count(≤ i)."""
    if not er_multiset:
        return 0.0

    r_values = []
    sorted_multiset = sorted(er_multiset)

    for i in range(1, n):
        # Count values at most i+1 and at most i
        count_le_i_plus_1 = sum(1 for val in sorted_multiset if val <= i + 1)
        count_le_i = sum(1 for val in sorted_multiset if val <= i)

        r_i = count_le_i_plus_1 - count_le_i
        r_values.append(r_i)

    return max(r_values) if r_values else 0.0


def analyze(path="./results"):
    """Analyze effective resistance density across graph sizes."""
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

    # Create scatter plot
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


if __name__ == "__main__":
    analyze()
