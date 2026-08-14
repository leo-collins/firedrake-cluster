import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


PLOT_DIR = Path(__file__).parent
RESULTS_DIR = PLOT_DIR.parent / "results"


def _numbered_values(row: dict[str, str], prefix: str) -> np.ndarray:
    values = sorted(
        (int(key.removeprefix(prefix)), float(value))
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


def _plot_apply(dofs_per_core: int, degree: int, kind: str, efficiency: bool):
    result_stem = f"contacting_apply_{kind}_3d_CG{degree}_{dofs_per_core}"
    data = get_data_apply(RESULTS_DIR / f"{result_stem}.csv")
    label = "matrix-free operator" if kind == "matfree" else "matrix"
    suffix = "_apply_efficiency.png" if efficiency else "_apply.png"
    title = f"Application of cross-mesh interpolation {label} between contacting domains\n(CG{degree}, {dofs_per_core:,} DoFs/process)"
    output_path = PLOT_DIR / "img" / f"{result_stem}{suffix}"
    if efficiency:
        _plot_weak_efficiency(data, output_path, title)
    else:
        _plot_timings(data, output_path, title)


def contacting_apply_matfree_3d_weakscaling(dofs_per_core: int, degree: int):
    _plot_apply(dofs_per_core, degree, "matfree", efficiency=False)


def contacting_apply_matfree_3d_weakscaling_efficiency(dofs_per_core: int, degree: int):
    _plot_apply(dofs_per_core, degree, "matfree", efficiency=True)


def contacting_apply_matrix_3d_weakscaling(dofs_per_core: int, degree: int):
    _plot_apply(dofs_per_core, degree, "matrix", efficiency=False)


def contacting_apply_matrix_3d_weakscaling_efficiency(dofs_per_core: int, degree: int):
    _plot_apply(dofs_per_core, degree, "matrix", efficiency=True)


def contacting_weakscaling_3d(dofs_per_core: int, degree: int):
    stem = f"contacting_weakscaling_3d_CG{degree}_{dofs_per_core}"
    data = get_data(RESULTS_DIR / f"{stem}.csv")
    _plot_timings(
        data,
        PLOT_DIR / "img" / f"{stem}.png",
        f"Cold-index cross-mesh matrix construction between contacting domains\n(CG{degree}, {dofs_per_core:,} DoFs/process)",
    )


def contacting_weakscaling_3d_efficiency(dofs_per_core: int, degree: int):
    stem = f"contacting_weakscaling_3d_CG{degree}_{dofs_per_core}"
    data = get_data(RESULTS_DIR / f"{stem}.csv")
    _plot_weak_efficiency(
        data,
        PLOT_DIR / "img" / f"{stem}_efficiency.png",
        f"Weak-scaling efficiency between contacting domains\n(CG{degree}, {dofs_per_core:,} DoFs/process)",
    )


if __name__ == "__main__":
    contacting_apply_matfree_3d_weakscaling(dofs_per_core=200_000, degree=3)
    contacting_apply_matfree_3d_weakscaling_efficiency(dofs_per_core=200_000, degree=3)
    contacting_apply_matrix_3d_weakscaling(dofs_per_core=200_000, degree=3)
    contacting_apply_matrix_3d_weakscaling_efficiency(dofs_per_core=200_000, degree=3)
    contacting_weakscaling_3d(dofs_per_core=200_000, degree=3)
    contacting_weakscaling_3d_efficiency(dofs_per_core=200_000, degree=3)
