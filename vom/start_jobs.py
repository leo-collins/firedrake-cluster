#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys


# Call .resolve() for compatibility with older versions of Python.
FILE_DIR = Path(__file__).parent.resolve()
SCRIPT_DIR = FILE_DIR / "scripts"
RESULTS_DIR = FILE_DIR / "results"
JOB_DIR = FILE_DIR / "jobs"
LOG_DIR = FILE_DIR / "logs"

JOB_TEMPLATE = """
#!/bin/bash

#PBS -N {job_name}
#PBS -l {resource_request}
#PBS -l place={exclusive}
#PBS -l walltime={wall_time}
#PBS -j oe
#PBS -o {log_dir}/

set -euo pipefail
export OMP_NUM_THREADS=1
export VOM_FLAMEGRAPH_SUBDIR="{result_subdir}"

cd "$PBS_O_WORKDIR"
cd "{script_dir}"

module load buildenv/default-foss-2025b
module load HDF5/1.14.6-gompi-2025b
module load Python/3.13.5-GCCcore-14.3.0

source "$HOME/{env}/venv-firedrake/bin/activate"

NPROCS={total_cpus}
POINT_COUNT={point_count}
CSV="{result_dir}/{script_name}_{point_count}.csv"

echo "Running {script_name}.py on $NPROCS processes."
echo "Firedrake environment: $HOME/{env}/venv-firedrake"
echo "POINT_COUNT=$POINT_COUNT."
echo "VertexOnlyMesh will be constructed with redundant=False."
echo "Results will be saved to $CSV."
echo "Job ID: $PBS_JOBID"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "CSV:    $CSV"
echo "PBS params: "
echo "  PBS -l {resource_request}"
echo "  PBS -l place={exclusive}"
echo "  PBS -l walltime={wall_time}"
echo "  PBS -j oe"
echo "  PBS -o {log_dir}/"

P={starting_proc}
while [ "$P" -le "$NPROCS" ]; do
    mpirun -n "$P" python {script_name}.py \
        {point_count} "$CSV" "$PBS_JOBID"
    P=$((P * 2))
done

echo "Job completed at $(date)."
"""


def check_script(script_name: str) -> bool:
    script_path = SCRIPT_DIR / f"{script_name}.py"
    return script_path.is_file()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and submit PBS jobs for the VOM benchmarks."
    )
    parser.add_argument(
        "script",
        help=(
            "Python script to run, relative to scripts/ and without the .py "
            "suffix."
        ),
    )
    parser.add_argument(
        "point_count",
        type=int,
        help=(
            "Points per core for weak scaling or total points for strong "
            "scaling."
        ),
    )
    parser.add_argument(
        "--result-subdir",
        default="distributed-rtree",
        help=(
            "Subdirectory below results/ for CSV output. "
            "Defaults to distributed-rtree."
        ),
    )
    parser.add_argument(
        "--env",
        choices=("firedrake-dev", "firedrake-dev2", "firedrake-dev3"),
        required=True,
        help=(
            "Firedrake environment directory below $HOME."
        ),
    )
    parser.add_argument(
        "--log-subdir",
        help="Optional subdirectory below logs/ for PBS output.",
    )
    parser.add_argument(
        "--ncpus",
        type=int,
        default=64,
        help="CPUs per node. On HX1, the maximum per node is 64.",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=4,
        help="Number of nodes to use. Defaults to 4.",
    )
    parser.add_argument(
        "--mem",
        type=int,
        default=400,
        help="Memory per node in GB. Defaults to 400 GB.",
    )
    parser.add_argument(
        "--range",
        action="store_true",
        help=(
            "Run powers of two from one process through ncpus*num_nodes. "
            "Without this flag, run only ncpus*num_nodes processes."
        ),
    )
    parser.add_argument(
        "--exclusive",
        action="store_true",
        help="Request exclusive access to the allocated nodes.",
    )
    parser.add_argument(
        "--walltime",
        type=int,
        default=240,
        help="Wall time in minutes. Defaults to 240 (4 hours).",
    )
    return parser.parse_args()


def get_time_str(minutes: int) -> str:
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}:00"


if __name__ == "__main__":
    args = parse_args()
    result_subdir = Path(args.result_subdir)
    if result_subdir.is_absolute() or ".." in result_subdir.parts:
        print("Error: --result-subdir must stay below the results directory.")
        sys.exit(2)
    result_dir = RESULTS_DIR / result_subdir
    log_subdir = Path(args.log_subdir) if args.log_subdir else Path()
    if log_subdir.is_absolute() or ".." in log_subdir.parts:
        print("Error: --log-subdir must stay below the logs directory.")
        sys.exit(2)
    log_dir = LOG_DIR / log_subdir

    if not check_script(args.script):
        print(f"Error: Script '{args.script}.py' not found in {SCRIPT_DIR}.")
        sys.exit(1)

    if args.ncpus < 1 or args.ncpus > 64:
        print("Error: ncpus per node must be between 1 and 64 on HX1.")
        sys.exit(1)

    if args.num_nodes < 1 or args.num_nodes > 32:
        print("Error: num_nodes must be between 1 and 32 on HX1.")
        sys.exit(1)

    if args.mem < 1 or args.mem > 400:
        print("Error: mem per node must be between 1 and 400 GB on HX1.")
        sys.exit(1)

    if args.point_count < 1:
        print("Error: point_count must be at least 1.")
        sys.exit(1)

    if args.walltime < 1:
        print("Error: walltime must be at least 1 minute.")
        sys.exit(1)

    total_cpus = args.ncpus * args.num_nodes
    job_name = f"{args.script}_{args.point_count}"

    result_dir.mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    job_script_map = {
        "job_name": job_name,
        "resource_request": (
            f"select={args.num_nodes}:ncpus={args.ncpus}:"
            f"mpiprocs={args.ncpus}:mem={args.mem}gb"
        ),
        "exclusive": "excl" if args.exclusive else "free",
        "wall_time": get_time_str(args.walltime),
        "starting_proc": 1 if args.range else total_cpus,
        "script_dir": SCRIPT_DIR,
        "total_cpus": total_cpus,
        "point_count": args.point_count,
        "env": args.env,
        "result_dir": result_dir,
        "result_subdir": result_subdir,
        "log_dir": log_dir,
        "script_name": args.script,
    }

    job_script = JOB_TEMPLATE.format_map(job_script_map)
    job_script_path = JOB_DIR / (
        f"{job_name}_{str(result_subdir).replace('/', '_')}_{total_cpus}.pbs"
    )
    try:
        with job_script_path.open("w") as f:
            f.write(job_script)
        print(f"Generated job script: {job_script_path}")
        print(f"Results directory: {result_dir}")
        subprocess.run(["qsub", str(job_script_path)], check=True)
    finally:
        # Remove the temporary script whether submission succeeds or fails.
        try:
            job_script_path.unlink()
        except FileNotFoundError:
            pass
