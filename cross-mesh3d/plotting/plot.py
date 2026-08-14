import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


PLOT_DIR = Path(__file__).parent
RESULTS_DIR = PLOT_DIR.parent / "results"


def _numbered_values(row: dict[str, str], prefix: str) -> np.ndarray:
    values = sorted(
        (
            int(key.removeprefix(prefix)),
            float(value),
        )
        for key, value in row.items()
        if key.startswith(prefix)
        and key.removeprefix(prefix).isdigit()
        and value not in {None, ""}
    )
    if not values:
        raise ValueError(f"No {prefix}N timing columns found in CSV row")
    return np.asarray([value for _, value in values], dtype=float)


def _get_data(csv_path: Path, prefix: str):
    with csv_path.open() as csv_file:
        rows = [
            {
                "nprocs": int(row["nprocs"]),
                "dofs_per_core": float(row["dofs_per_core"]),
                "times": _numbered_values(row, prefix),
            }
            for row in csv.DictReader(csv_file)
        ]
    return sorted(rows, key=lambda row: row["nprocs"])


def get_data(csv_path: Path):
    return _get_data(csv_path, "run")


def get_data_apply(csv_path: Path):
    return _get_data(csv_path, "apply")


def _summary(data):
    nprocs = np.asarray([row["nprocs"] for row in data], dtype=int)
    medians = np.asarray([np.median(row["times"]) for row in data])
    q1 = np.asarray([np.percentile(row["times"], 25) for row in data])
    q3 = np.asarray([np.percentile(row["times"], 75) for row in data])
    return nprocs, medians, q1, q3


def _finish(fig, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_timings(data, output_path: Path, title: str):
    nprocs, medians, q1, q3 = _summary(data)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(nprocs, medians, marker="o", label="Median")
    ax.fill_between(nprocs, q1, q3, alpha=0.2, label="Interquartile range")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylim(0, None)
    ax.set_xlabel("Number of MPI processes")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--")
    ax.legend()
    _finish(fig, output_path)


def _plot_weak_efficiency(data, output_path: Path, title: str):
    nprocs, medians, _, _ = _summary(data)
    efficiency = medians[0] / medians
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(nprocs, efficiency, marker="o")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Number of MPI processes")
    ax.set_ylabel("Weak-scaling efficiency")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--")
    _finish(fig, output_path)


def _plot_strong_scaling(data, output_path: Path, title: str, efficiency: bool):
    nprocs, medians, _, _ = _summary(data)
    relative_processes = nprocs / nprocs[0]
    speedup = medians[0] / medians

    fig, ax = plt.subplots(figsize=(8, 6))
    if efficiency:
        values = speedup / relative_processes
        ax.plot(nprocs, values, marker="o")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Strong-scaling efficiency")
    else:
        ax.plot(nprocs, speedup, marker="o", label="Measured")
        ax.plot(nprocs, relative_processes, "--", color="black", alpha=0.7, label="Ideal")
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylabel("Speedup")
        ax.legend()

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Number of MPI processes")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--")
    _finish(fig, output_path)


def overlapping_weakscaling_3d(dofs_per_core: int, degree: int):
    stem = f"overlapping_weakscaling_3d_CG{degree}_{dofs_per_core}"
    data = get_data(RESULTS_DIR / f"{stem}.csv")
    _plot_timings(
        data,
        PLOT_DIR / "img" / f"{stem}.png",
        f"Weak scaling of cold-index cross-mesh matrix construction\n(CG{degree}, {dofs_per_core:,} DoFs/process)",
    )


def overlapping_weakscaling_3d_efficiency(dofs_per_core: int, degree: int):
    stem = f"overlapping_weakscaling_3d_CG{degree}_{dofs_per_core}"
    data = get_data(RESULTS_DIR / f"{stem}.csv")
    _plot_weak_efficiency(
        data,
        PLOT_DIR / "img" / f"{stem}_efficiency.png",
        f"Weak-scaling efficiency of cold-index cross-mesh matrix construction\n(CG{degree}, {dofs_per_core:,} DoFs/process)",
    )


def overlapping_strongscaling_3d_speedup(total_dofs: int, degree: int):
    stem = f"overlapping_strongscaling_3d_CG{degree}_{total_dofs}"
    data = get_data(RESULTS_DIR / f"{stem}.csv")
    _plot_strong_scaling(
        data,
        PLOT_DIR / "img" / f"{stem}_speedup.png",
        f"Strong scaling of cold-index cross-mesh matrix construction\n(CG{degree}, requested DoFs={total_dofs:,})",
        efficiency=False,
    )


def overlapping_strongscaling_3d_efficiency(total_dofs: int, degree: int):
    stem = f"overlapping_strongscaling_3d_CG{degree}_{total_dofs}"
    data = get_data(RESULTS_DIR / f"{stem}.csv")
    _plot_strong_scaling(
        data,
        PLOT_DIR / "img" / f"{stem}_efficiency.png",
        f"Strong-scaling efficiency of cold-index cross-mesh matrix construction\n(CG{degree}, requested DoFs={total_dofs:,})",
        efficiency=True,
    )


def _plot_apply(degree: int, dofs: int, kind: str, efficiency: bool):
    stem = f"overlapping_apply_{kind}_3d_CG{degree}_{dofs}"
    data = get_data_apply(RESULTS_DIR / f"{stem}.csv")
    label = "matrix-free operator" if kind == "matfree" else "matrix"
    output_suffix = "_apply_efficiency.png" if efficiency else "_apply.png"
    output_stem = f"overlapping_apply_{kind}_3d_CG{degree}_{dofs}"
    title = f"Application of cross-mesh interpolation {label}\n(CG{degree}, {dofs:,} DoFs/process)"
    if efficiency:
        _plot_weak_efficiency(data, PLOT_DIR / "img" / f"{output_stem}{output_suffix}", title)
    else:
        _plot_timings(data, PLOT_DIR / "img" / f"{output_stem}{output_suffix}", title)


def overlapping_apply_matfree_3d_weakscaling(degree: int, dofs: int):
    _plot_apply(degree, dofs, "matfree", efficiency=False)


def overlapping_apply_matfree_3d_weakscaling_efficiency(degree: int, dofs: int):
    _plot_apply(degree, dofs, "matfree", efficiency=True)


def overlapping_apply_matrix_3d_weakscaling(degree: int, dofs: int):
    _plot_apply(degree, dofs, "matrix", efficiency=False)


def overlapping_apply_matrix_3d_weakscaling_efficiency(degree: int, dofs: int):
    _plot_apply(degree, dofs, "matrix", efficiency=True)


def overlapping_weakscaling_oneform_3d(dofs_per_core: int, degree: int):
    stem = f"overlapping_weakscaling_one_form_3d_CG{degree}_{dofs_per_core}"
    data = get_data(RESULTS_DIR / f"{stem}.csv")
    output_stem = f"overlapping_weakscaling_oneform_3d_CG{degree}_{dofs_per_core}"
    _plot_timings(
        data,
        PLOT_DIR / "img" / f"{output_stem}.png",
        f"Weak scaling of cold-index cross-mesh interpolation\n(CG{degree}, {dofs_per_core:,} DoFs/process)",
    )


def overlapping_weakscaling_oneform_3d_efficiency(dofs_per_core: int, degree: int):
    stem = f"overlapping_weakscaling_one_form_3d_CG{degree}_{dofs_per_core}"
    data = get_data(RESULTS_DIR / f"{stem}.csv")
    output_stem = f"overlapping_weakscaling_oneform_3d_CG{degree}_{dofs_per_core}"
    _plot_weak_efficiency(
        data,
        PLOT_DIR / "img" / f"{output_stem}_efficiency.png",
        f"Weak-scaling efficiency of cold-index cross-mesh interpolation\n(CG{degree}, {dofs_per_core:,} DoFs/process)",
    )


if __name__ == "__main__":
    overlapping_apply_matfree_3d_weakscaling(degree=3, dofs=200_000)
    overlapping_apply_matfree_3d_weakscaling_efficiency(degree=3, dofs=200_000)
    overlapping_apply_matrix_3d_weakscaling(degree=3, dofs=200_000)
    overlapping_apply_matrix_3d_weakscaling_efficiency(degree=3, dofs=200_000)
    overlapping_weakscaling_3d(dofs_per_core=200_000, degree=3)
    overlapping_weakscaling_3d_efficiency(dofs_per_core=200_000, degree=3)
    overlapping_strongscaling_3d_speedup(total_dofs=10_000_000, degree=3)
    overlapping_strongscaling_3d_efficiency(total_dofs=10_000_000, degree=3)
    overlapping_weakscaling_oneform_3d(dofs_per_core=200_000, degree=3)
    overlapping_weakscaling_oneform_3d_efficiency(dofs_per_core=200_000, degree=3)
