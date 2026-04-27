#!/usr/bin/env python3
import argparse
from html import parser
from pathlib import Path
import os
import sys

SCRIPT_DIR = Path(__file__).parent / "scripts"
RESULT_DIR = SCRIPT_DIR / "results"
JOB_DIR = Path(__file__).parent / "jobs"

JOB_TEMPLATE = """
#!/bin/bash

# PBS -N {job_name}
# PBS -l select={num_nodes}:ncpus={cpus_per_node}:mpiprocs={cpus_per_node}:mem={mem_per_node}gb
# PBS -l place={exclusive}
# PBS -l walltime={wall_time}
# PBS -j oe
# PBS -o logs/

set -euo pipefail
export OMP_NUM_THREADS=1

cd $PBS_O_WORKDIR
cd {script_dir}

module load buildenv/default-foss-2025b
module load HDF5/1.14.6-gompi-2025b
module load Python/3.13.5-GCCcore-14.3.0

source "$HOME/firedrake-dev/venv-firedrake/bin/activate"

NPROCS={total_cpus}
DOFS={dof_count}
DEGREE={degree}
CSV="{result_dir}/{script_name}_CG{degree}_{dof_count}.csv"

echo "Running {script_name}.py with DOFs=$DOFS, degree=$DEGREE, on $NPROCS processes."
echo "Results will be saved to $CSV."
echo "Job ID: $PBS_JOBID"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "CSV:    $CSV"
echo "PBS params: "
echo "  PBS -l select={num_nodes}:ncpus={cpus_per_node}:mpiprocs={cpus_per_node}:mem={mem_per_node}gb"
echo "  PBS -l place={exclusive}"
echo "  PBS -l walltime={wall_time}"
echo "  PBS -j oe"
echo "  PBS -o logs/"

P={starting_proc}
while [ "$P" -le "$NPROCS" ]; do
    mpirun -n "$P" python {script_name}.py {dof_count} {degree} "$CSV"
    P=$((P * 2))
done

echo "Job completed at $(date)."
"""

def check_script(script_name: str) -> bool:
    # checks if the script exists
    script_name += ".py"
    script_path = SCRIPT_DIR / script_name
    if not script_path.is_file():
        return False
    else:
        return True

def parse_args():
    parser = argparse.ArgumentParser(description="Generate PBS job scripts for Firedrake cluster runs.")
    parser.add_argument("script", type=str, 
                        help="The Python script to run (relative to the scripts directory). For example, 'run_experiment' will look for 'scripts/run_experiment.py'.")
    parser.add_argument("dof_count", type=int, 
                        help="Dofs per core (for weak scaling) or total dofs (for strong scaling).")
    parser.add_argument("degree", type=int, help="Degree of the CG element.")
    parser.add_argument("--ncpus", type=int, default=64, 
                        help="CPUs per node to run. On HX1, the maximum per node is 64.")
    parser.add_argument("--num_nodes", type=int, default=4, help="Number of nodes to use. Defaults to 4.")
    parser.add_argument("--mem", type=int, default=400, help="Memory per node in GB. Defaults to 400GB.")
    parser.add_argument("--range", action="store_true", default=False, help="If set, run range of jobs in powers of 2 from 1 up to the total number of CPUs (ncpus * num_nodes). If not set, only run the job with the total number of CPUs.")
    parser.add_argument("--exclusive", action="store_true", default=False, help="If set, request exclusive access to nodes.")
    parser.add_argument("--walltime", type=int, default="240", help="Wall time for the job in minutes. Defaults to 240 minutes (4 hours).")
    return parser.parse_args()

def get_time_str(minutes: int) -> str:
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}:00"

if __name__ == "__main__":
    args = parse_args()

    if not check_script(args.script):
        print(f"Error: Script '{args.script}.py' not found in {SCRIPT_DIR}.")
        sys.exit(1)
    
    if args.ncpus > 64:
        print("Error: ncpus per node cannot exceed 64 on HX1.")
        sys.exit(1)
    
    if args.num_nodes > 32:
        print("Error: num_nodes cannot exceed 32 on HX1.")
        sys.exit(1)
    
    if args.mem > 400:
        print("Error: mem per node cannot exceed 400GB on HX1.")
        sys.exit(1)
    
    total_cpus = args.ncpus * args.num_nodes

    job_name = f"{args.script}_CG{args.degree}_{args.dof_count}"

    job_script_map = {
        "job_name": job_name,
        "num_nodes": args.num_nodes,
        "cpus_per_node": args.ncpus,
        "mem_per_node": args.mem,
        "exclusive": "excl" if args.exclusive else "free",
        "wall_time": get_time_str(args.walltime),
        "starting_proc": 1 if args.range else total_cpus,
        "script_dir": SCRIPT_DIR,
        "total_cpus": total_cpus,
        "dof_count": args.dof_count,
        "degree": args.degree,
        "result_dir": RESULT_DIR,
        "script_name": args.script,
    }

    job_script = JOB_TEMPLATE.format_map(job_script_map)
    job_script_path = JOB_DIR / f"{job_name}.pbs"
    with open(job_script_path, "w") as f:
        f.write(job_script)
        print(f"Generated job script: {job_script_path}")
    
    # Make logs directory if it doesn't exist
    logs_dir = JOB_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Start the job
    os.system(f"qsub {job_script_path}")

    # delete the job script after submission
    os.remove(job_script_path)
    


