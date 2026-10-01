"""Plot every cross-mesh scaling CSV in results/<branch>/.

Run with ``python plot.py``; figures are written to plotting/img/.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUTPUT = HERE / "img"
CORES_PER_NODE = 64


def plot_csv(path):
    timings = defaultdict(list)
    with path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            runs = np.array([
                float(row[f"run{i}"]) for i in range(10) if row.get(f"run{i}")
            ])
            runs = runs[np.isfinite(runs) & (runs > 0)]
            if not len(runs):
                continue
            q1, q3 = np.percentile(runs, [25, 75])
            spread = q3 - q1
            timings[int(row["nprocs"])].extend(
                runs[(runs >= q1 - 1.5 * spread) & (runs <= q3 + 1.5 * spread)]
            )

    if not timings:
        return
    cores = np.array(sorted(timings))
    median = np.array([np.median(timings[n]) for n in cores])
    weak = "weakscaling" in path.stem
    degree = path.stem.split("_CG")[-1].split("_")[0]
    size = int(path.stem.rsplit("_", 1)[-1])
    problem = f"CG{degree}, {size:,} {'DoFs/core' if weak else 'total DoFs'}"
    branch = path.parent.name
    OUTPUT.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cores, median, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xticks(cores, [str(n) for n in cores])
    ax.set_xlabel("Number of cores")
    ax.set_ylabel("Median run time (s)")
    ax.set_title(f"{'Weak' if weak else 'Strong'} scaling of cross-mesh interpolation matrix assembly\n"
                 f"{branch} ({problem})")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{branch}_{path.stem}.png", dpi=300)
    plt.close(fig)

    if weak:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        panels = ((cores <= CORES_PER_NODE, "Intranode efficiency"),
                  (cores >= CORES_PER_NODE, "Internode efficiency"))
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]
        panels = ((np.ones(len(cores), dtype=bool), "Strong scaling efficiency"),)

    for ax, (mask, label) in zip(axes, panels):
        selected_cores = cores[mask]
        ax.set_title(label)
        if len(selected_cores):
            baseline = median[mask][0]
            factor = 1 if weak else selected_cores[0] / selected_cores
            efficiency = baseline / median[mask] * factor
            ax.plot(selected_cores, efficiency, marker="o")
            ax.set_xscale("log", base=2)
            ax.set_xticks(selected_cores, [str(n) for n in selected_cores])
        else:
            ax.text(0.5, 0.5, "No internode results", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_xlabel("Number of cores")
        ax.grid(True, linestyle="--", alpha=0.6)
    axes[0].set_ylabel("Efficiency")
    fig.suptitle(f"{'Weak' if weak else 'Strong'} scaling efficiency of cross-mesh interpolation assembly\n"
                 f"{branch} ({problem})")
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{branch}_{path.stem}_efficiency.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    for csv_path in sorted(RESULTS.glob("*/*.csv")):
        plot_csv(csv_path)
