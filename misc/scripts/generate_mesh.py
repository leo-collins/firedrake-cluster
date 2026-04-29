import warnings
import sys
from pathlib import Path
from time import perf_counter_ns
from math import floor, ceil
warnings.filterwarnings("ignore")
from firedrake import *

dofs_per_core = int(sys.argv[1])
degree = int(sys.argv[2])
n_meshes = int(sys.argv[3])
n_cores = COMM_WORLD.size

mesh_dir = Path(__file__).parent.parent / "meshes" 
mesh_dir.mkdir(exist_ok=True)
mesh_path = mesh_dir / f"mesh{n_meshes}_{n_cores}procs_{dofs_per_core}dofs_CG{degree}.h5"

n = max(floor((((dofs_per_core * n_cores) ** (1 / 3)) - 1) / degree), 1)
PETSc.Sys.Print(f"n={n}")

t0 = perf_counter_ns()
if n_meshes == 1:
    mesh1 = UnitCubeMesh(n, n, n, name="mesh1")
else:
    mesh2 = UnitCubeMesh(ceil(1.01*n), ceil(1.01*n), ceil(1.01*n), name="mesh2")
t1 = perf_counter_ns()
mesh_gen_time_s = (t1 - t0) / 1e9
PETSc.Sys.Print(f"nprocs={n_cores}: mesh generation={mesh_gen_time_s:.6g}s")

# Save meshes to file for later use

with CheckpointFile(str(mesh_path), "w") as f:
    if n_meshes == 1:
        f.save_mesh(mesh1)
    else:
        f.save_mesh(mesh2)
PETSc.Sys.Print(f"nprocs={n_cores}: saved meshes to {mesh_path}")