import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


CATEGORIES = (
    ("Omega_v_time_s", r"$\Omega_v$"),
    ("Omega_v_io_time_s", r"$\Omega_v$ I/O"),
    ("A_time_s", "A"),
    ("B_time_s", "B"),
    ("C_time_s", "C"),
    ("I_time_s", "I"),
)


def median_timing(row: dict[str, str], prefix: str) -> float:
    run_times = np.array([
        float(value)
        for key, value in row.items()
        if key.startswith(f"{prefix}_run") and value != ""
    ])
    return float(np.median(run_times))


def get_data(csv_path: Path):
    data = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            timings = {
                key: median_timing(row, key)
                for key, _ in CATEGORIES
            }
            data.append({
                "nprocs": int(row["nprocs"]),
                "degree": int(row["degree"]),
                "dofs_per_core": float(row["dofs_per_core"]),
                "timings": timings,
            })
    return sorted(data, key=lambda d: d["nprocs"])


def plot_operator_timings(csv_path: Path):
    output_path = Path(__file__).parent / "plotting" / "img" / "operator_timings_oneform.png"
    data = get_data(csv_path)

    nprocs = [d["nprocs"] for d in data]
    degree = data[0]["degree"]
    dofs_per_core = int(round(data[0]["dofs_per_core"]))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    for key, label in CATEGORIES:
        median_run_times = [d["timings"][key] for d in data]
        ax.plot(nprocs, median_run_times, marker="o", label=label)

    ax.legend()
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylim(0, None)
    ax.set_xlabel("Number of MPI processes")
    ax.set_ylabel("Median wall time (s)")
    ax.set_title(f"One-form cross-mesh operator timings \n (CG{degree}, {dofs_per_core:,} DoFs/process)")
    ax.grid(True, which="both", ls="--")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)


def plot_operator_efficiency(csv_path: Path):
    output_path = Path(__file__).parent / "plotting" / "img" / "operator_timings_oneform_efficiency.png"
    data = get_data(csv_path)

    nprocs = [d["nprocs"] for d in data]
    degree = data[0]["degree"]
    dofs_per_core = int(round(data[0]["dofs_per_core"]))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    for key, label in CATEGORIES:
        median_run_times = np.array([d["timings"][key] for d in data])
        efficiency = median_run_times[0] / median_run_times
        ax.plot(nprocs, efficiency, marker="o", label=label)

    ax.legend()
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Number of MPI processes")
    ax.set_ylabel("Efficiency")
    ax.set_title(f"One-form cross-mesh operator timing efficiency \n (CG{degree}, {dofs_per_core:,} DoFs/process)")
    ax.grid(True, which="both", ls="--")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)


if __name__ == "__main__":
    csv_path = Path(__file__).parent / "time_operator_oneform.csv"
    plot_operator_timings(csv_path)
    plot_operator_efficiency(csv_path)
