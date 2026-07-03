from sys import exit
import subprocess
from tqdm import tqdm

def cmd_boltzmann(size, seed):
    return [
        "python3",
        "planar_graph_generator.py",
        "--method=boltzmann",
        f"--n-vertices={size}",
        f"--seed={seed}",
    ]


def cmd_delaunay(size, seed):
    return [
        "python3",
        "planar_graph_generator.py",
        "--method=delaunay",
        f"--n-vertices={size}",
        f"--seed={seed}",
    ]


GRAPH_SIZES = [8, 64]
N_SEEDS = 30

try:
    for size in GRAPH_SIZES:
        print(f"\nGenerating graphs with N={size}")

        with tqdm(
            total=N_SEEDS,
            desc=f"N={size}",
            unit="graph",
        ) as pbar:

            for seed in range(N_SEEDS):

                subprocess.run(
                    cmd_boltzmann(size, seed),
                    check=True,
                )

                # Uncomment for second generator:
                # subprocess.run(
                #     cmd_delaunay(size, seed),
                #     check=True,
                # )

                pbar.update(1)

except KeyboardInterrupt:
    print("\nInterrupted. Exiting immediately.")
    exit(130)
