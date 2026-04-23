import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PLOT_MASS_MATRIX = True

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

def plot_weakscaling(dofs_per_core: int, degree: int, job_id: str, one_form: bool = False):
    if one_form:
        prefix = "overlapping_weakscaling_oneform"
    else:
        prefix = "overlapping_weakscaling"
    output_path = Path(__file__).parent / "img" / f"{prefix}_3d_CG{degree}_{dofs_per_core}_{job_id}.png"
    csv_path = Path(__file__).parent.parent / "scripts" / "results" / f"{prefix}_3d_CG{degree}_{dofs_per_core}_{job_id}.csv"
    data = get_data(csv_path)
    
    nprocs = [d["nprocs"] for d in data]
    run_times = np.array([[d["run1"], d["run2"], d["run3"]] for d in data])
    average_run_times = np.mean(run_times, axis=1)

    plt.figure()
    plt.plot(nprocs, average_run_times, marker="o")
    # plot line at 64 cores
    plt.axvline(x=64, color="red", linestyle="--", label="64 cores", alpha=0.3)
    plt.legend()
    plt.xscale("log", base=2)
    plt.ylim(0, None)
    plt.xlabel("Number of processes")
    plt.ylabel("Average run time (s)")
    plt.title(f"Weak scaling of cross-mesh interpolation matrix assembly \n (CG{degree}, {dofs_per_core} dofs/core)")
    plt.grid(True, which="both", ls="--")
    plt.savefig(output_path, dpi=300)

def plot_weakscaling_efficiency(dofs_per_core: int, degree: int, job_id: str, one_form: bool = False):
    if one_form:
        prefix = "overlapping_weakscaling_oneform"
    else:
        prefix = "overlapping_weakscaling"
    output_path = Path(__file__).parent / "img" / f"{prefix}_3d_CG{degree}_{dofs_per_core}_{job_id}_efficiency.png"
    csv_path = Path(__file__).parent.parent / "scripts" / "results" / f"{prefix}_3d_CG{degree}_{dofs_per_core}_{job_id}.csv"
    data = get_data(csv_path)
    
    nprocs = [d["nprocs"] for d in data]
    run_times = np.array([[d["run1"], d["run2"], d["run3"]] for d in data])
    average_run_times = np.mean(run_times, axis=1)

    efficiency = average_run_times[0] / average_run_times

    plt.figure()
    plt.plot(nprocs, efficiency, marker="o")
    # plot perfect efficiency line
    plt.plot(nprocs, np.ones_like(nprocs), linestyle="--", label="Perfect efficiency", color="black", alpha=0.7)
    # plot line at 64 cores
    plt.axvline(x=64, color="red", linestyle="--", label="64 cores", alpha=0.3)
    plt.legend()
    plt.xscale("log", base=2)
    plt.xlabel("Number of processes")
    plt.ylabel("Efficiency")
    plt.ylim(0, 1.05)
    plt.title(f"Weak scaling efficiency of cross-mesh interpolation assembly \n (CG{degree}, {dofs_per_core} dofs/core)")
    plt.grid(True, which="both", ls="--")
    plt.savefig(output_path, dpi=300)

def plot_strongscaling_speedup(total_dofs: int, degree: int, job_id: str):
    output_path = Path(__file__).parent / "img" / f"overlapping_strongscaling_3d_CG{degree}_{total_dofs}_{job_id}_speedup.png"
    csv_path = Path(__file__).parent.parent / "scripts" / "results" / f"overlapping_strongscaling_3d_CG{degree}_{total_dofs}_{job_id}.csv"
    data = get_data(csv_path)
    
    nprocs = [d["nprocs"] for d in data]
    # exclude run0 as it includes setup costs
    run_times = np.array([[d["run1"], d["run2"], d["run3"]] for d in data])
    average_run_times = np.mean(run_times, axis=1)

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

if __name__ == "__main__":
    # strong scaling CG3 10M dofs
    plot_strongscaling_speedup(10000000, 3, "542212.pbs-6")
    plot_strongscaling_loglog(10000000, 3, "542212.pbs-6")

    # strong scaling CG3 1M dofs
    plot_strongscaling_speedup(1000000, 3, "542214.pbs-6")
    plot_strongscaling_loglog(1000000, 3, "542214.pbs-6")

    # weak scaling CG3 1M dofs/core
    plot_weakscaling(1000000, 3, "542222.pbs-6")
    plot_weakscaling_efficiency(1000000, 3, "542222.pbs-6")

    # weak scaling CG3 100k dofs/core
    plot_weakscaling(100000, 3, "542223.pbs-6")
    plot_weakscaling_efficiency(100000, 3, "542223.pbs-6")

    # weak scaling CG3 10k dofs/core
    plot_weakscaling(10000, 3, "542225.pbs-6")
    plot_weakscaling_efficiency(10000, 3, "542225.pbs-6")

    # plot weak scaling oneform CG3 10k dofs/core
    plot_weakscaling(10000, 3, "542236.pbs-6", one_form=True)
    plot_weakscaling_efficiency(10000, 3, "542236.pbs-6", one_form=True)

    # plot weak scaling oneform CG3 100k dofs/core
    plot_weakscaling(100000, 3, "542235.pbs-6", one_form=True)
    plot_weakscaling_efficiency(100000, 3, "542235.pbs-6", one_form=True)

    # plot weak scaling oneform CG3 1M dofs/core
    plot_weakscaling(1000000, 3, "542233.pbs-6", one_form=True)
    plot_weakscaling_efficiency(1000000, 3, "542233.pbs-6", one_form=True)