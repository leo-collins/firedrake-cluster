import numpy as np
from firedrake import *
from time import perf_counter_ns
from mpi4py import MPI
from sys import argv
from pathlib import Path
import csv
import warnings
warnings.filterwarnings("ignore")

# Run with:
#   mpiexec -n <nprocs> python vom_simple.py [csv_path]
# if csv_path is not given, results will only be printed to stdout. 

csv_path = Path(argv[1]) if len(argv) > 1 else None

PETSc.Sys.Print(f"nranks={COMM_WORLD.size}")

mesh = UnitSquareMesh(100, 100)
PETSc.Sys.Print(f"Finished creating mesh")

rng = np.random.default_rng(COMM_WORLD.rank)
points = rng.random((100_000, 2))

if COMM_WORLD.size == 1:
    # Warm up cache
    VertexOnlyMesh(mesh, points, redundant=False)

t0 = perf_counter_ns()
vom = VertexOnlyMesh(mesh, points, redundant=False)
t1 = perf_counter_ns()

if COMM_WORLD.rank == 0:
    print(f"Rank 0 time {(t1 - t0) / 1e9}s")
avg_time_s = COMM_WORLD.allreduce(t1 - t0, op=MPI.SUM) / (COMM_WORLD.size * 1e9)
PETSc.Sys.Print(f"Avg. time: {avg_time_s} s")

if COMM_WORLD.rank == 0 and csv_path is not None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "nprocs",
                "time_s",
            ],
        )
        if write_header:
            w.writeheader()
        w.writerow(
            {
                "nprocs": COMM_WORLD.size,
                "time_s": avg_time_s,
            }
        )