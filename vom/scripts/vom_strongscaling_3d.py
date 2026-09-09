import csv
from pathlib import Path
from sys import argv
from time import perf_counter_ns

import numpy as np
from firedrake import COMM_WORLD, PETSc, UnitCubeMesh, VertexOnlyMesh
from firedrake.petsc import garbage_cleanup
from mpi4py import MPI


# Run with:
#   mpiexec -n <nprocs> python vom_strongscaling_3d.py \
#       <total_points> [csv_path] [pbs_job_id] [n_runs]

if len(argv) < 2:
    raise ValueError(
        "Usage: vom_strongscaling_3d.py "
        "<total_points> [csv_path] [pbs_job_id] [n_runs]"
    )

nprocs = COMM_WORLD.size
total_points = int(argv[1])
csv_path = Path(argv[2]) if len(argv) > 2 else None
pbs_job_id = argv[3] if len(argv) > 3 else None
n_runs = int(argv[4]) if len(argv) > 4 else 10

if n_runs < 1:
    raise ValueError("n_runs must be >= 1")

if total_points < nprocs:
    raise ValueError(
        "total_points must be at least the number of MPI processes"
    )

mesh_n = 162
total_mesh_cells = 6 * mesh_n**3
PETSc.Sys.Print(
    f"nprocs={nprocs}: constructing UnitCubeMesh({mesh_n}, {mesh_n}, {mesh_n}) "
    f"with {total_mesh_cells} tetrahedra"
)
mesh = UnitCubeMesh(mesh_n, mesh_n, mesh_n)
PETSc.Sys.Print(f"nprocs={nprocs}: mesh construction complete")
mesh.tolerance = 0.5
total_mesh_vertices = (mesh_n + 1) ** 3

# Split the fixed global point count as evenly as possible across ranks.
local_points = total_points // nprocs
if COMM_WORLD.rank < total_points % nprocs:
    local_points += 1
first_point = COMM_WORLD.rank * (total_points // nprocs)
first_point += min(COMM_WORLD.rank, total_points % nprocs)
bit_generator = np.random.PCG64(0)
bit_generator.advance(3 * first_point)
rng = np.random.Generator(bit_generator)
points = rng.random((local_points, 3))

# Warm up Firedrake/PyOP2 compilation before collecting timings.
vom = VertexOnlyMesh(mesh, points, redundant=False)

times = []
for run in range(n_runs):
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
    ] + [f"run{i}" for i in range(n_runs)]
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
            "points_per_core": total_points / nprocs,
            "total_points": total_points,
            "mean_time_s": mean_time_s,
            "std_time_s": std_time_s,
        }
        row.update({f"run{i}": elapsed_s for i, elapsed_s in enumerate(times)})
        writer.writerow(row)
