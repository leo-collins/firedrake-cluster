from math import ceil, floor
from sys import argv
from time import perf_counter_ns
import warnings

warnings.filterwarnings("ignore")

from mpi4py import MPI
from benchmark_utils import reset_cross_mesh_caches
from firedrake import *
from firedrake.utility_meshes import _mark_mesh_boundaries

# This tests weak parallel scaling of assembly of cross-mesh interpolation
# matrices with fully overlapping meshes.
# Run with:
#   mpiexec -n <nprocs> python overlapping_flamegraph3d.py <dofs_per_core> <degree> -log_view :foo.txt:ascii_flamegraph

if len(argv) < 3:
    raise ValueError("Usage: overlapping_flamegraph3d.py <dofs_per_core> <degree>")

n_cores = COMM_WORLD.size
dofs_per_core = int(argv[1])
degree = int(argv[2])
if degree < 1:
    raise ValueError("degree must be >= 1")

# For UnitCubeMesh, dim(CG(degree)) = (degree * n + 1)^3.
n = max(floor((((dofs_per_core * n_cores) ** (1 / 3)) - 1) / degree), 1)

# meshes have different number of nodes to force different parallel partitions
mesh1 = UnitCubeMesh(n, n, n)
mesh2 = UnitCubeMesh(ceil(1.01 * n), ceil(1.01 * n), ceil(1.01 * n))
PETSc.Sys.Print("Meshes created")

V = FunctionSpace(mesh1, "CG", degree)
W = FunctionSpace(mesh2, "CG", degree)

interp = interpolate(TrialFunction(V), W)

# Warm generated code before profiling, then restore cold construction state.
warmup_matrix = assemble(interp, mat_type="aij")
del warmup_matrix

reset_cross_mesh_caches(interp, mesh1)
COMM_WORLD.barrier()
with PETSc.Log.Event("run0"):
    t0 = perf_counter_ns()
    assemble(interp, mat_type="aij")
    t1 = perf_counter_ns()

t = COMM_WORLD.allreduce(t1 - t0, op=MPI.MAX) / 1e9
PETSc.Sys.Print(f"run0: {t:.6f} s")

reset_cross_mesh_caches(interp, mesh1)
COMM_WORLD.barrier()
with PETSc.Log.Event("run1"):
    t0 = perf_counter_ns()
    assemble(interp, mat_type="aij")
    t1 = perf_counter_ns()

t = COMM_WORLD.allreduce(t1 - t0, op=MPI.MAX) / 1e9
PETSc.Sys.Print(f"run1: {t:.6f} s")
