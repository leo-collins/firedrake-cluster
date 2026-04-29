import warnings
import psutil
import sys
from time import perf_counter_ns
from math import floor, ceil
warnings.filterwarnings("ignore")
from firedrake import *

dofs_per_core = 5000
degree = 1
n_cores = COMM_WORLD.size

n = max(floor((((dofs_per_core * n_cores) ** (1 / 3)) - 1) / degree), 1)
PETSc.Sys.Print(f"n={n}")

t0 = perf_counter_ns()
mesh1 = UnitCubeMesh(n, n, n)
mesh2 = UnitCubeMesh(ceil(1.01*n), ceil(1.01*n), ceil(1.01*n))
t1 = perf_counter_ns()
mesh_gen_time_s = (t1 - t0) / 1e9
PETSc.Sys.Print(f"nprocs={n_cores}: mesh generation={mesh_gen_time_s:.6g}s")

V = FunctionSpace(mesh1, "CG", degree)
W = FunctionSpace(mesh2, "CG", degree)

PETSc.Sys.Print(f"Avergae dofs per core: {(W.dim() + V.dim()) / (2 * n_cores):.6g}")

# print memory usage statistics
process = psutil.Process()
mem_info = process.memory_info()
if COMM_WORLD.rank == 0:
    print(f"nprocs={n_cores}: memory usage: RSS={mem_info.rss / 1e9:.6g}GB, VMS={mem_info.vms / 1e9:.6g}GB")