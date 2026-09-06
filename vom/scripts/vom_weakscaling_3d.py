import csv
from pathlib import Path
from sys import argv
from time import perf_counter_ns

import numpy as np
from firedrake import COMM_WORLD, PETSc, UnitCubeMesh, VertexOnlyMesh
from firedrake.petsc import garbage_cleanup
from mpi4py import MPI


# Run with:
#   mpiexec -n <nprocs> python vom_weakscaling_3d.py \
#       <points_per_core> [csv_path] [pbs_job_id]

if len(argv) < 2:
    raise ValueError(
        "Usage: vom_weakscaling_3d.py "
        "<points_per_core> [csv_path] [pbs_job_id]"
    )

nprocs = COMM_WORLD.size
points_per_core = int(argv[1])
csv_path = Path(argv[2]) if len(argv) > 2 else None
pbs_job_id = argv[3] if len(argv) > 3 else None

if points_per_core < 1:
    raise ValueError("points_per_core must be >= 1")

# Keep approximately 25,000 mesh cells per core.
target_mesh_cells_per_core = 25_000
mesh_n = max(round((target_mesh_cells_per_core * nprocs / 6) ** (1 / 3)), 1)
total_mesh_cells = 6 * mesh_n**3
PETSc.Sys.Print(
    f"nprocs={nprocs}: constructing UnitCubeMesh({mesh_n}, {mesh_n}, {mesh_n}) "
    f"with {total_mesh_cells} tetrahedra"
)
mesh = UnitCubeMesh(mesh_n, mesh_n, mesh_n)
PETSc.Sys.Print(f"nprocs={nprocs}: mesh construction complete")
mesh.tolerance = 0.5
total_mesh_vertices = (mesh_n + 1) ** 3

bit_generator = np.random.PCG64(0)
bit_generator.advance(3 * COMM_WORLD.rank * points_per_core)
rng = np.random.Generator(bit_generator)
points = rng.random((points_per_core, 3))
total_points = points_per_core * nprocs

# Warm up Firedrake/PyOP2 compilation before collecting timings.
vom = VertexOnlyMesh(mesh, points, redundant=False)

times = []
for run in range(10):
    del vom
    mesh._partition_rtree_cache = None
    mesh._rtree_cache = None
    mesh._box_ratio_heuristic_cache = None
    mesh._bounding_box_coords_cache = None
    garbage_cleanup(mesh)

    COMM_WORLD.barrier()
    t0 = perf_counter_ns()
    vom = VertexOnlyMesh(mesh, points, redundant=False)
    t1 = perf_counter_ns()
    elapsed_s = COMM_WORLD.allreduce(t1 - t0, op=MPI.MAX) / 1e9
    times.append(elapsed_s)
    PETSc.Sys.Print(f"nprocs={nprocs}: run {run} time={elapsed_s:.6f}s")

mean_time_s = np.mean(times)
std_time_s = np.std(times)
PETSc.Sys.Print(f"Average VOM creation time: {mean_time_s:.6f}s")
PETSc.Sys.Print(f"Standard deviation: {std_time_s:.6f}s")

del vom
garbage_cleanup(mesh)

if COMM_WORLD.rank == 0 and csv_path is not None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    fieldnames = [
        "nprocs",
        "pbs_job_id",
        "mesh_n",
        "mesh_cells_per_core",
        "total_mesh_cells",
        "mesh_vertices_per_core",
        "total_mesh_vertices",
        "points_per_core",
        "total_points",
        "mean_time_s",
        "std_time_s",
    ] + [f"run{i}" for i in range(10)]
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {
            "nprocs": nprocs,
            "pbs_job_id": pbs_job_id,
            "mesh_n": mesh_n,
            "mesh_cells_per_core": total_mesh_cells / nprocs,
            "total_mesh_cells": total_mesh_cells,
            "mesh_vertices_per_core": total_mesh_vertices / nprocs,
            "total_mesh_vertices": total_mesh_vertices,
            "points_per_core": points_per_core,
            "total_points": total_points,
            "mean_time_s": mean_time_s,
            "std_time_s": std_time_s,
        }
        row.update({f"run{i}": elapsed_s for i, elapsed_s in enumerate(times)})
        writer.writerow(row)
