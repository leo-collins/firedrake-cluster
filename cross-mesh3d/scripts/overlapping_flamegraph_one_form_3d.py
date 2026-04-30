from math import ceil, floor
from sys import argv
import warnings
warnings.filterwarnings("ignore")

from firedrake import *
from firedrake.utility_meshes import _mark_mesh_boundaries

# This tests weak parallel scaling of assembly of cross-mesh interpolation
# matrices with fully overlapping meshes.
# Run with:
#   mpiexec -n <nprocs> python overlapping_flamegraph3d.py <dofs_per_core> <degree> -log_view :foo.txt:ascii_flamegraph

if len(argv) < 2:
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
f = Function(V).assign(1.1)
interp = interpolate(f, W)

with PETSc.Log.Event("run0"):
    assemble(interp)

del interp._interpolator

with PETSc.Log.Event("run1"):
    assemble(interp)
