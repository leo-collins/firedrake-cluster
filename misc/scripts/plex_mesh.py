import warnings
import sys
from pathlib import Path
import psutil
from time import perf_counter_ns
from math import floor, ceil
warnings.filterwarnings("ignore")
from firedrake import *
from firedrake.utility_meshes import _mark_mesh_boundaries, _refine_quads_to_triangles

dofs_per_core = int(sys.argv[1])
degree = int(sys.argv[2])
n_cores = COMM_WORLD.size

n = max(floor((((dofs_per_core * n_cores) ** (1 / 3)) - 1) / degree), 1)
PETSc.Sys.Print(f"n={n}")
mesh = BoxMesh(2, 2, 2, 1.0, 1.0, 1.0, hexahedral=True)

t0 = perf_counter_ns()
plex1 = PETSc.DMPlex().createBoxMesh(
    faces=(n, n, n),
    lower=(0.0, 0.0, 0.0),
    upper=(1.0, 1.0, 1.0),
    comm=COMM_WORLD,
    simplex=False,
)
_mark_mesh_boundaries(plex1)
plex1 = _refine_quads_to_triangles(plex1, "left")
plex2 = PETSc.DMPlex().createBoxMesh(
    faces=(ceil(1.01*n), ceil(1.01*n), ceil(1.01*n)),
    lower=(0.0, 0.0, 0.0),
    upper=(1.0, 1.0, 1.0),
    comm=COMM_WORLD,
    simplex=False,
)
_mark_mesh_boundaries(plex2)
plex2 = _refine_quads_to_triangles(plex2, "left")
mesh1 = Mesh(plex1, name="mesh1")
mesh2 = Mesh(plex2, name="mesh2")
t1 = perf_counter_ns()
mesh_gen_time_s = (t1 - t0) / 1e9
PETSc.Sys.Print(f"nprocs={n_cores}: mesh generation={mesh_gen_time_s:.6g}s")

V = FunctionSpace(mesh1, "CG", degree)
W = FunctionSpace(mesh2, "CG", degree)

average_dofs_per_core = (W.dim() + V.dim()) / (2 * n_cores)
PETSc.Sys.Print(f"nprocs={n_cores}: n={n}, degree={degree}, average dofs/core={average_dofs_per_core:.6g}")

# print memory usage statistics
process = psutil.Process()
mem_info = process.memory_info()
if COMM_WORLD.rank == 0:
    print(f"nprocs={n_cores}: memory usage: RSS={mem_info.rss / 1e9:.6g}GB, VMS={mem_info.vms / 1e9:.6g}GB")