import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from pathlib import Path


def get_data(csv_path: Path):
    data = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "nprocs": int(row["nprocs"]),
                "dofs_per_core": float(row["dofs_per_core"]),
                "run0": float(row["run0"]),
                "run1": float(row["run1"]),
                "run2": float(row["run2"]),
                "run3": float(row["run3"]),
            })
    return data

def get_data_apply(csv_path: Path):
    data = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "nprocs": int(row["nprocs"]),
                "dofs_per_core": float(row["dofs_per_core"]),
                "apply0": float(row["apply0"]),
                "apply1": float(row["apply1"]),
                "apply2": float(row["apply2"]),
                "apply3": float(row["apply3"]),
                "apply4": float(row["apply4"]),
                "apply5": float(row["apply5"]),
            })
    return data

def plot_weakscaling(dofs_per_core: int, degree: int):
    output_path = Path(__file__).parent / "img" / f"overlapping_weakscaling_3d_CG{degree}_{dofs_per_core}.png"
    csv_path = Path(__file__).parent.parent / "scripts" / "results" / f"overlapping_weakscaling_3d_CG{degree}_{dofs_per_core}.csv"
    data = get_data(csv_path)
    
    nprocs = []
    average_run_times = []
    for d in data:
        run_times = np.array([d["run0"], d["run1"], d["run2"], d["run3"]])
        average_run_time = np.mean(np.sort(run_times)[:2])  # take average of fastest 2 runs
        average_run_times.append(average_run_time)
        nprocs.append(d["nprocs"])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(nprocs, average_run_times, marker="o")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylim(0, None)
    ax.set_xlabel("Number of cores")
    ax.set_ylabel("Average run time (s)")
    ax.set_title(f"Weak scaling of cross-mesh interpolation matrix assembly \n (CG{degree}, {dofs_per_core} dofs/core)")
    ax.grid(True, which="both", ls="--")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)


def plot_weakscaling_efficiency(dofs_per_core: int, degree: int):
    output_path = Path(__file__).parent / "img" / f"overlapping_weakscaling_3d_CG{degree}_{dofs_per_core}.png"
    csv_path = Path(__file__).parent.parent / "scripts" / "results" / f"overlapping_weakscaling_3d_CG{degree}_{dofs_per_core}_efficiency.csv"
    data = get_data(csv_path)
    
    nprocs = []
    average_run_times = []
    for d in data:
        run_times = np.array([d["run0"], d["run1"], d["run2"], d["run3"]])
        average_run_time = np.mean(np.sort(run_times)[:2])  # take average of fastest 2 runs
        average_run_times.append(average_run_time)
        nprocs.append(d["nprocs"])

    # finish when runs are complete

def plot_strongscaling_speedup(total_dofs: int, degree: int):
    output_path = Path(__file__).parent / "img" / f"overlapping_strongscaling_3d_CG{degree}_{total_dofs}_speedup.png"
    csv_path = Path(__file__).parent.parent / "scripts" / "results" / f"overlapping_strongscaling_3d_CG{degree}_{total_dofs}.csv"
    data = get_data(csv_path)
    
    nprocs = []
    average_run_times = []
    for d in data:
        run_times = np.array([d["run0"], d["run1"], d["run2"], d["run3"]])
        average_run_time = np.mean(np.sort(run_times)[:2])  # take average of fastest 2 runs
        average_run_times.append(average_run_time)
        nprocs.append(d["nprocs"])

    speedup = average_run_times[0] / average_run_times
    perfect_speedup = np.array(nprocs) / nprocs[0]

    plt.figure()
    plt.plot(nprocs, speedup, marker="o")
    plt.plot(nprocs, perfect_speedup, linestyle="--", label="Perfect speedup", color="black", alpha=0.7)
    plt.legend()
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel("Number of processes")
    plt.ylabel("Speedup")
    plt.title(f"Strong scaling speedup of cross-mesh interpolation matrix assembly \n (CG{degree}, total dofs={total_dofs:,})")
    plt.grid(True, which="both", ls="--")
    plt.savefig(output_path, dpi=300)


def plot_strongscaling_efficiency(total_dofs: int, degree: int, job_id: str):
    output_path = Path(__file__).parent / "img" / f"overlapping_strongscaling_3d_CG{degree}_{total_dofs}_{job_id}_efficiency.png"
    csv_path = Path(__file__).parent.parent / "scripts" / "results" / f"overlapping_strongscaling_3d_CG{degree}_{total_dofs}_{job_id}.csv"
    data = get_data(csv_path)
    
    nprocs = [d["nprocs"] for d in data]
    run_times = np.array([[d["run0"], d["run1"], d["run2"], d["run3"]] for d in data])
    average_run_times = np.mean(run_times, axis=1)

    speedup = average_run_times[0] / average_run_times
    efficiency = speedup / nprocs

    plt.figure()
    plt.plot(nprocs, efficiency, marker="o")
    # plt.xscale("log", base=2)
    plt.xlabel("Number of processes")
    plt.ylabel("Efficiency")
    plt.title("Strong scaling efficiency of cross-mesh interpolation assembly")
    plt.grid(True, which="both", ls="--")
    plt.savefig(output_path, dpi=300)


def plot_strongscaling_loglog(total_dofs: int, degree: int, job_id: str):
    output_path = Path(__file__).parent / "img" / f"overlapping_strongscaling_3d_CG{degree}_{total_dofs}_{job_id}.png"
    csv_path = Path(__file__).parent.parent / "scripts" / "results" / f"overlapping_strongscaling_3d_CG{degree}_{total_dofs}_{job_id}.csv"
    data = get_data(csv_path)
    
    nprocs = [d["nprocs"] for d in data]
    # exclude run0 as it includes setup costs
    run_times = np.array([[d["run1"], d["run2"], d["run3"]] for d in data])
    average_run_times = np.mean(run_times, axis=1)

    plt.figure()
    plt.plot(nprocs, average_run_times, marker="o")
    # Plot theoretical perfect scaling line
    perfect_run_times = average_run_times[0] / np.array(nprocs)
    plt.plot(nprocs, perfect_run_times, linestyle="--", label="Perfect scaling", color="black", alpha=0.7)
    plt.legend()
    plt.xscale("log", base=2)
    plt.yscale("log", base=10)
    plt.xlabel("Number of ranks")
    plt.ylabel("Average run time (s)")
    plt.title(f"Strong scaling of cross-mesh interpolation matrix assembly \n (CG{degree}, total dofs={total_dofs:,})")

    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

def overlapping_apply_matfree_3d_weakscaling(degree: int, dofs: int):
    output_path = Path(__file__).parent / "img" / f"overlapping_apply_matfree_3d_CG{degree}_{dofs}_apply.png"
    csv_path = Path(__file__).parent.parent / "results" / f"overlapping_apply_matfree_3d_CG{degree}_{dofs}.csv"
    data = get_data_apply(csv_path)
    
    nprocs = []
    average_run_times = []
    for d in data:
        run_times = np.array([d["apply0"], d["apply1"], d["apply2"], d["apply3"], d["apply4"], d["apply5"]])
        average_run_time = np.mean(np.sort(run_times)[:3])  # take average of fastest 3 runs
        average_run_times.append(average_run_time)
        nprocs.append(d["nprocs"])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(nprocs, average_run_times, marker="o")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylim(0, None)
    ax.set_xlabel("Number of cores")
    ax.set_ylabel("Average run time (s)")
    ax.set_title(f"Application of matfree cross-mesh interpolation operator \n (CG{degree}, dofs per core={dofs:,})")
    ax.grid(True, which="both", ls="--")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)

def overlapping_apply_matfree_3d_weakscaling_efficiency(degree: int, dofs: int):
    output_path = Path(__file__).parent / "img" / f"overlapping_apply_matfree_3d_CG{degree}_{dofs}_apply_efficiency.png"
    csv_path = Path(__file__).parent.parent / "results" / f"overlapping_apply_matfree_3d_CG{degree}_{dofs}.csv"
    data = get_data_apply(csv_path)
    
    nprocs = []
    average_run_times = []
    for d in data:
        run_times = np.array([d["apply0"], d["apply1"], d["apply2"], d["apply3"], d["apply4"], d["apply5"]])
        average_run_time = np.mean(np.sort(run_times)[:3])  # take average of fastest 3 runs
        average_run_times.append(average_run_time)
        nprocs.append(d["nprocs"])

    intranode_efficiency = average_run_times[0] / average_run_times[:7]
    internode_efficiency = average_run_times[6] / average_run_times[6:]

    fig = plt.figure(figsize=(10, 6))
    ax1, ax2 = fig.subplots(1, 2, sharey=True)
    # Plot intranode efficiency
    ax1.plot(nprocs[:7], intranode_efficiency, marker="o")
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Number of cores")
    ax1.set_ylabel("Efficiency")
    ax1.set_title("Intranode efficiency")
    ax1.grid(True, which="both", ls="--")
    # Plot internode efficiency
    ax2.plot(nprocs[6:], internode_efficiency, marker="o")
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Number of cores")
    ax2.set_title("Internode efficiency")
    ax2.grid(True, which="both", ls="--")
    fig.suptitle(f"Application of matfree cross-mesh interpolation operator \n (CG{degree}, dofs per core={dofs:,})")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)


def overlapping_apply_matrix_3d_weakscaling(degree: int, dofs: int):
    output_path = Path(__file__).parent / "img" / f"overlapping_apply_matrix_3d_CG{degree}_{dofs}_apply.png"
    csv_path = Path(__file__).parent.parent / "results" / f"overlapping_apply_matrix_3d_CG{degree}_{dofs}.csv"
    data = get_data_apply(csv_path)
    
    nprocs = []
    average_run_times = []
    for d in data:
        run_times = np.array([d["apply0"], d["apply1"], d["apply2"], d["apply3"], d["apply4"], d["apply5"]])
        average_run_time = np.mean(np.sort(run_times)[:3])  # take average of fastest 3 runs
        average_run_times.append(average_run_time)
        nprocs.append(d["nprocs"])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(nprocs, average_run_times, marker="o")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylim(0, None)
    ax.set_xlabel("Number of cores")
    ax.set_ylabel("Average run time (s)")
    ax.set_title(f"Application of cross-mesh interpolation matrix \n (CG{degree}, dofs per core={dofs:,})")
    ax.grid(True, which="both", ls="--")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)

def overlapping_apply_matrix_3d_weakscaling_efficiency(degree: int, dofs: int):
    output_path = Path(__file__).parent / "img" / f"overlapping_apply_matrix_3d_CG{degree}_{dofs}_apply_efficiency.png"
    csv_path = Path(__file__).parent.parent / "results" / f"overlapping_apply_matrix_3d_CG{degree}_{dofs}.csv"
    data = get_data_apply(csv_path)
    
    nprocs = []
    average_run_times = []
    for d in data:
        run_times = np.array([d["apply0"], d["apply1"], d["apply2"], d["apply3"], d["apply4"], d["apply5"]])
        average_run_time = np.mean(np.sort(run_times)[:3])  # take average of fastest 3 runs
        average_run_times.append(average_run_time)
        nprocs.append(d["nprocs"])

    intranode_efficiency = average_run_times[0] / average_run_times[:7]
    internode_efficiency = average_run_times[6] / average_run_times[6:]

    fig = plt.figure(figsize=(10, 6))
    ax1, ax2 = fig.subplots(1, 2, sharey=True)
    # Plot intranode efficiency
    ax1.plot(nprocs[:7], intranode_efficiency, marker="o")
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Number of cores")
    ax1.set_ylabel("Efficiency")
    ax1.set_title("Intranode efficiency")
    ax1.grid(True, which="both", ls="--")
    # Plot internode efficiency
    ax2.plot(nprocs[6:], internode_efficiency, marker="o")
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Number of cores")
    ax2.set_title("Internode efficiency")
    ax2.grid(True, which="both", ls="--")
    fig.suptitle(f"Application of cross-mesh interpolation matrix \n (CG{degree}, dofs per core={dofs:,})")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)

if __name__ == "__main__":
    overlapping_apply_matfree_3d_weakscaling(degree=3, dofs=200_000)
    overlapping_apply_matfree_3d_weakscaling_efficiency(degree=3, dofs=200_000)
    overlapping_apply_matrix_3d_weakscaling(degree=3, dofs=200_000)
    overlapping_apply_matrix_3d_weakscaling_efficiency(degree=3, dofs=200_000)