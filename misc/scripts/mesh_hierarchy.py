import sys
import psutil
import warnings
from time import perf_counter_ns
warnings.filterwarnings("ignore", category=UserWarning, module="h5py")
from firedrake import *

dofs_per_core = int(sys.argv[1])
degree = int(sys.argv[2])
n_cores = COMM_WORLD.size

refinements = 3
scale = 2 ** refinements
coarse_scale = 2 ** (refinements - 1)
target = float(dofs_per_core)
base_n = max(int(((2 * target * n_cores) / (degree ** 3 * (scale ** 3 + coarse_scale ** 3))) ** (1 / 3)), 1)

while True:
	finest_dofs = (degree * base_n * scale + 1) ** 3
	coarser_dofs = (degree * base_n * coarse_scale + 1) ** 3
	average = (finest_dofs + coarser_dofs) / (2 * n_cores)
	if average >= target:
		if base_n > 1:
			prev_finest_dofs = (degree * (base_n - 1) * scale + 1) ** 3
			prev_coarser_dofs = (degree * (base_n - 1) * coarse_scale + 1) ** 3
			prev_average = (prev_finest_dofs + prev_coarser_dofs) / (2 * n_cores)
			if average - target > target - prev_average:
				base_n -= 1
				continue
		break
	base_n += 1

t0 = perf_counter_ns()
mesh = UnitCubeMesh(base_n, base_n, base_n)
hierarchy = MeshHierarchy(mesh, refinements)
t1 = perf_counter_ns()
mesh_gen_time_s = (t1 - t0) / 1e9
PETSc.Sys.Print(f"nprocs={n_cores}: mesh generation={mesh_gen_time_s:.6g}s")
mesh1 = hierarchy[-1]
mesh2 = hierarchy[-2]

V = FunctionSpace(mesh1, "CG", degree)
W = FunctionSpace(mesh2, "CG", degree)

PETSc.Sys.Print(f"nprocs={n_cores}: base_n={base_n}, refinements={refinements}")
PETSc.Sys.Print(
	f"nprocs={n_cores}: target dofs/core={dofs_per_core:g}, degree={degree}, achieved average dofs/core={(V.dim() + W.dim()) / (2 * n_cores):.6g}"
)