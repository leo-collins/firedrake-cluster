import csv
from math import floor, sqrt
from pathlib import Path
from sys import argv
from time import perf_counter_ns
import warnings
warnings.filterwarnings("ignore")

from firedrake import *
from mpi4py import MPI

# This tests weak parallel scaling of assembly of a mass matrix
# Run with:
#   mpiexec -n <nprocs> python overlapping_weakscaling.py <dofs_per_core> [csv_path]

n_cores = COMM_WORLD.size
dofs_per_core = int(argv[1]) if len(argv) > 1 else 50_000  # default 50k dofs/core
csv_path = Path(argv[2]) if len(argv) > 2 else None
n = int(floor(sqrt(dofs_per_core * n_cores) - 1))  # works for CG1 UnitSquareMesh

# meshes have different number of nodes to force different parallel partitions
t0_mesh = perf_counter_ns()
mesh1 = UnitSquareMesh(n, n)
mesh2 = UnitSquareMesh(n + 1, n + 1)
t1_mesh = perf_counter_ns()
mesh_gen_time_s = (t1_mesh - t0_mesh) / 1e9
PETSc.Sys.Print(f"nprocs={n_cores}: mesh generation={mesh_gen_time_s:.6g}s")

V = FunctionSpace(mesh1, "CG", 1)
W = FunctionSpace(mesh2, "CG", 1)

mass = inner(TrialFunction(V), TestFunction(W)) * dx

if n_cores == 1:
    # Warm up cache
    assemble(mass, mat_type="aij")

t0 = perf_counter_ns()
assemble(mass, mat_type="aij")
t1 = perf_counter_ns()

avg_time_s = COMM_WORLD.allreduce(t1 - t0, op=MPI.SUM) / (n_cores * 1e9)
actual_dofs_per_core = V.dim() / n_cores

if COMM_WORLD.rank == 0:
    PETSc.Sys.Print(
        f"nprocs={n_cores}: assembly={avg_time_s:.6g}s"
    )
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "nprocs",
                    "dofs_per_core",
                    "mesh_gen_time_s",
                    "assembly_time_s",
                ],
            )
            if write_header:
                w.writeheader()
            w.writerow(
                {
                    "nprocs": n_cores,
                    "dofs_per_core": actual_dofs_per_core,
                    "mesh_gen_time_s": mesh_gen_time_s,
                    "assembly_time_s": avg_time_s,
                }
            )
