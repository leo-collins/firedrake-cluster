import warnings
import sys
from pathlib import Path
from time import perf_counter_ns
from math import floor, ceil
warnings.filterwarnings("ignore")
from firedrake import *

dofs_per_core = int(sys.argv[1])
degree = int(sys.argv[2])
n_cores = COMM_WORLD.size

mesh_dir = Path(__file__).parent.parent / "meshes"
mesh_path = mesh_dir / f"mesh_{n_cores}procs_{dofs_per_core}dofs_CG{degree}.h5"

t0 = perf_counter_ns()
with CheckpointFile(str(mesh_path), "r") as f:
    mesh1 = f.load_mesh("mesh1")
    mesh2 = f.load_mesh("mesh2")
t1 = perf_counter_ns()
mesh_gen_time_s = (t1 - t0) / 1e9
PETSc.Sys.Print(f"nprocs={n_cores}: load meshes={mesh_gen_time_s:.6g}s")

V = FunctionSpace(mesh1, "CG", degree)
W = FunctionSpace(mesh2, "CG", degree)

PETSc.Sys.Print(f"Average dofs per core: {(W.dim() + V.dim()) / (2 * n_cores):.6g}")
