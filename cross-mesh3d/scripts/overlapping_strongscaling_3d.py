import csv
from math import floor, ceil
from pathlib import Path
from sys import argv
from time import perf_counter_ns
import warnings
warnings.filterwarnings("ignore")

from firedrake import *
from mpi4py import MPI

# This tests parallel scaling of assembly of cross-mesh interpolation
# matrices with fully overlapping meshes.
# Run with:
#   mpiexec -n <nprocs> python overlapping_strongscaling_3d.py <total_dofs> <degree> [csv_path]

if len(argv) < 3:
	raise ValueError("Usage: overlapping_strongscaling_3d.py <total_dofs> <degree> [csv_path]")

n_cores = COMM_WORLD.size
total_dofs = int(argv[1])
degree = int(argv[2])
if degree < 1:
	raise ValueError("degree must be >= 1")
csv_path = Path(argv[3]) if len(argv) > 3 else None

# For UnitCubeMesh, dim(CG(degree)) = (degree * n + 1)^3.
n = max(int(floor(((total_dofs ** (1 / 3)) - 1) / degree)), 1)

# meshes have different number of nodes to force different parallel partitions
t0_mesh = perf_counter_ns()
mesh1 = UnitCubeMesh(n, n, n)
mesh2 = UnitCubeMesh(ceil(1.01*n), ceil(1.01*n), ceil(1.01*n))
t1_mesh = perf_counter_ns()
mesh_gen_time_s = (t1_mesh - t0_mesh) / 1e9
PETSc.Sys.Print(f"nprocs={n_cores}: mesh generation={mesh_gen_time_s:.6g}s")

V = FunctionSpace(mesh1, "CG", degree)
W = FunctionSpace(mesh2, "CG", degree)

interp = interpolate(TrialFunction(V), W)

run_times_s = []
for _ in range(4):
	COMM_WORLD.barrier()
	t0 = perf_counter_ns()
	assemble(interp, mat_type="aij")
	t1 = perf_counter_ns()
	run_time_s = COMM_WORLD.allreduce(t1 - t0, op=MPI.MAX) / 1e9
	PETSc.Sys.Print(f"nprocs={n_cores}: run time={run_time_s:.6g}s")
	run_times_s.append(run_time_s)
	# delete cached interpolator (which includes cached VOM)
	del interp._interpolator

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
					"run0",
					"run1",
					"run2",
					"run3",
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
					"run0": run_times_s[0],
					"run1": run_times_s[1],
					"run2": run_times_s[2],
					"run3": run_times_s[3],
				}
			)
