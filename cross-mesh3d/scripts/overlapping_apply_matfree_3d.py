import csv
import warnings
from math import ceil, floor
from pathlib import Path
from sys import argv
from time import perf_counter_ns

warnings.filterwarnings("ignore")

from mpi4py import MPI

from firedrake import *

# This benchmarks the matrix-free cross-mesh interpolation operator assembly
# and timing of its application.
# Run with:
#   mpiexec -n <nprocs> python overlapping_matfree_3d.py <dofs_per_core> <degree> [csv_path]

if len(argv) < 3:
    raise ValueError(
        "Usage: overlapping_matfree_3d.py <dofs_per_core> <degree> [csv_path]"
    )

n_cores = COMM_WORLD.size
dofs_per_core = int(argv[1])
degree = int(argv[2])
if degree < 1:
    raise ValueError("degree must be >= 1")
csv_path = Path(argv[3]) if len(argv) > 3 else None

# For UnitCubeMesh, dim(CG(degree)) = (degree * n + 1)^3.
n = max(floor((((dofs_per_core * n_cores) ** (1 / 3)) - 1) / degree), 1)

# meshes have different number of nodes to force different parallel partitions
t0_mesh = perf_counter_ns()
mesh1 = UnitCubeMesh(n, n, n)
mesh2 = UnitCubeMesh(ceil(1.01 * n), ceil(1.01 * n), ceil(1.01 * n))
t1_mesh = perf_counter_ns()
mesh_gen_time_s = (t1_mesh - t0_mesh) / 1e9
PETSc.Sys.Print(f"nprocs={n_cores}: mesh generation={mesh_gen_time_s:.6g}s")

V = FunctionSpace(mesh1, "CG", degree)
W = FunctionSpace(mesh2, "CG", degree)

# Create function to apply interpolation to
u = Function(V).assign(1.1)

# Assemble matrix-free operator
interp = interpolate(TrialFunction(V), W)
t0 = perf_counter_ns()
I = assemble(interp, mat_type="matfree")
# apply once to build VOM
assemble(I @ u)
t1 = perf_counter_ns()
assembly_time_s = COMM_WORLD.allreduce(t1 - t0, op=MPI.MAX) / 1e9
PETSc.Sys.Print(
    f"nprocs={n_cores}: initial assembly time={assembly_time_s:.6g}s"
)

apply_times_s = []

for run_idx in range(4):
    COMM_WORLD.barrier()
    t0 = perf_counter_ns()
    res = assemble(I @ u)
    t1 = perf_counter_ns()
    apply_time_s = COMM_WORLD.allreduce(t1 - t0, op=MPI.MAX) / 1e9
    apply_times_s.append(apply_time_s)
    PETSc.Sys.Print(f"nprocs={n_cores}: run{run_idx} apply time={apply_time_s:.6g}s")

average_dofs_per_core = (W.dim() + V.dim()) / (2 * n_cores)

if COMM_WORLD.rank == 0:
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "nprocs",
                    "degree",
                    "dofs_per_core",
                    "mesh_gen_time_s",
                    "apply0",
                    "apply1",
                    "apply2",
                    "apply3",
                ],
            )
            if write_header:
                w.writeheader()
            w.writerow(
                {
                    "nprocs": n_cores,
                    "degree": degree,
                    "dofs_per_core": average_dofs_per_core,
                    "mesh_gen_time_s": mesh_gen_time_s,
                    "apply0": apply_times_s[0],
                    "apply1": apply_times_s[1],
                    "apply2": apply_times_s[2],
                    "apply3": apply_times_s[3],
                }
            )
