import argparse
import csv
import fcntl
from pathlib import Path
from sys import argv, path
from time import perf_counter_ns

parser = argparse.ArgumentParser(description="Benchmark VOM creation on a unit cube.")
parser.add_argument("points_per_core", type=int)
parser.add_argument("csv_path", nargs="?", type=Path)
parser.add_argument("pbs_job_id", nargs="?")
parser.add_argument("n_runs", nargs="?", type=int, default=10)
parser.add_argument("--high-order", action="store_true", help="Use a warped cubic Netgen mesh.")
args = parser.parse_args()
# PETSc initializes on import; the benchmark arguments have already been parsed.
argv[:] = argv[:1]

import numpy as np
from benchmark_utils import _firedrake_git_branch
from firedrake import (
    COMM_WORLD, PETSc, DirichletBC, Function, Mesh, MeshHierarchy,
    SpatialCoordinate, VectorFunctionSpace, VertexOnlyMesh, as_vector, pi, sin,
)
from firedrake.mesh import VertexOnlyMeshSF
from firedrake.petsc import garbage_cleanup
from mpi4py import MPI

path.insert(0, str(Path(__file__).resolve().parents[1] / "meshes"))
from generate_cube_hierarchy import refined_cube


# Run with:
#   mpiexec -n <nprocs> python vom.py \
#       <points_per_core> [csv_path] [pbs_job_id] [n_runs] [--high-order]


def high_order_cube(target_cells, levels):
    """Refine a Netgen cube in parallel and warp its cubic coordinates."""
    from netgen.csg import unit_cube
    from netgen.meshing import Mesh as NetgenMesh

    target_seed_cells = target_cells / 8**levels
    maxh = (6 / target_seed_cells) ** (1 / 3)
    ngmesh = NetgenMesh(dim=3)
    if COMM_WORLD.rank == 0:
        for attempt in range(6):
            ngmesh = unit_cube.GenerateMesh(maxh=maxh)
            if abs(ngmesh.ne / target_seed_cells - 1) <= 0.1 or attempt == 5:
                break
            maxh *= (ngmesh.ne / target_seed_cells) ** (1 / 3)
    maxh = COMM_WORLD.bcast(maxh, root=0)
    seed = Mesh(ngmesh)
    # A fresh coordinate Function avoids Netgen's CAD snapping on refinement.
    seed = Mesh(seed.coordinates.copy())
    mesh = MeshHierarchy(seed, levels)[-1]

    h = maxh / 2**levels
    periods = max(1, round(1 / (4 * h)))
    x = SpatialCoordinate(mesh)
    waves = [sin(2 * pi * periods * x[i]) for i in range(3)]
    displacement = as_vector([
        0.2 * h * sin(pi * x[i]) * waves[(i + 1) % 3] * waves[(i + 2) % 3]
        for i in range(3)
    ])
    V = VectorFunctionSpace(mesh, "CG", 3)
    coordinates = Function(V).interpolate(x + displacement)
    DirichletBC(V, x, "on_boundary").apply(coordinates)
    return Mesh(coordinates)


nprocs = COMM_WORLD.size
points_per_core = args.points_per_core
csv_path = args.csv_path
pbs_job_id = args.pbs_job_id
n_runs = args.n_runs

if points_per_core < 1:
    raise ValueError("points_per_core must be >= 1")
if n_runs < 1:
    raise ValueError("n_runs must be >= 1")

# Keep approximately 25,000 mesh cells per core.
target_mesh_cells_per_core = 25_000
levels = 3
mesh_type = "high order" if args.high_order else "linear"
if args.high_order:
    mesh = high_order_cube(target_mesh_cells_per_core * nprocs, levels)
else:
    mesh_n = max(round((target_mesh_cells_per_core * nprocs / 6) ** (1 / 3)), 1)
    # Three refinements require rounding the requested n to a multiple of eight.
    scale = 2 ** levels
    seed_n = max((mesh_n + scale // 2) // scale, 1)
    mesh = refined_cube(seed_n, levels)
total_mesh_cells = COMM_WORLD.allreduce(mesh.cell_set.size, op=MPI.SUM)
PETSc.Sys.Print(
    f"nprocs={nprocs}: {mesh_type} mesh construction complete, "
    f"{total_mesh_cells} tetrahedra, {total_mesh_cells / nprocs:.1f} cells/rank"
)
mesh.tolerance = 0.5
firedrake_git_branch = COMM_WORLD.bcast(
    _firedrake_git_branch() if COMM_WORLD.rank == 0 else None,
    root=0,
)

bit_generator = np.random.PCG64(0)
bit_generator.advance(3 * COMM_WORLD.rank * points_per_core)
rng = np.random.Generator(bit_generator)
points = rng.random((points_per_core, 3))

# Retain the candidate SF from the most recent VOM construction so its graph
# can be summarized after the timed call completes.
candidate_observation = None
original_candidate_sf = VertexOnlyMeshSF.candidate_sf


def recording_candidate_sf(cls, parent_mesh, root_coordinates):
    global candidate_observation
    candidate_observation = original_candidate_sf(parent_mesh, root_coordinates)
    return candidate_observation


VertexOnlyMeshSF.candidate_sf = classmethod(recording_candidate_sf)

# Warm up Firedrake/PyOP2 compilation before collecting timings.
vom = VertexOnlyMesh(mesh, points, redundant=False)

times = []
final_rank_diagnostics = None
for run in range(n_runs):
    del vom
    mesh._partition_rtree_cache = None
    mesh._rtree_cache = None
    mesh._box_ratio_heuristic_cache = None
    mesh._bounding_box_coords_cache = None
    garbage_cleanup(mesh)

    COMM_WORLD.barrier()
    t0 = perf_counter_ns()
    candidate_observation = None
    vom = VertexOnlyMesh(mesh, points, redundant=False)
    t1 = perf_counter_ns()
    local_time_s = (t1 - t0) / 1e9
    elapsed_s = COMM_WORLD.allreduce(t1 - t0, op=MPI.MAX) / 1e9
    times.append(elapsed_s)
    PETSc.Sys.Print(f"nprocs={nprocs}: run {run} time={elapsed_s:.6f}s")

    if candidate_observation is None:
        raise RuntimeError("VertexOnlyMesh did not construct a candidate SF")
    boxes = mesh._box_ratio_heuristic
    rank_diagnostics = {
        "candidate_leaves": candidate_observation.nleaves,
        "candidate_input_peers": np.unique(candidate_observation.input_ranks).size,
        "partition_boxes": boxes.shape[0],
        "local_time_s": local_time_s,
        "vom_owned_points": vom.cell_set.size,
    }
    if run == n_runs - 1:
        final_rank_diagnostics = COMM_WORLD.gather(rank_diagnostics, root=0)

mean_time_s = np.mean(times)
std_time_s = np.std(times)
PETSc.Sys.Print(f"Average VOM creation time: {mean_time_s:.6f}s")
PETSc.Sys.Print(f"Standard deviation: {std_time_s:.6f}s")

del vom
garbage_cleanup(mesh)

if COMM_WORLD.rank == 0 and csv_path is not None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_output = {}
    for name in final_rank_diagnostics[0]:
        values = np.asarray([row[name] for row in final_rank_diagnostics])
        diagnostic_output[f"{name}_min"] = values.min()
        diagnostic_output[f"{name}_median"] = np.median(values)
        diagnostic_output[f"{name}_max"] = values.max()
        if name == "partition_boxes":
            diagnostic_output["partition_boxes_total"] = values.sum()
    fieldnames = [
        "nprocs",
        "pbs_job_id",
        "firedrake_git_branch",
        "mesh_type",
        "mesh_cells_per_core",
        "points_per_core",
    ] + list(diagnostic_output)
    with csv_path.open("a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        existing_header = next(csv.reader(f), None)
        if existing_header is not None and existing_header != fieldnames:
            raise ValueError(
                f"CSV header does not match the benchmark columns: {csv_path}. "
                "Use a new CSV path or move the existing file before running."
            )
        f.seek(0, 2)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        row = {
            "nprocs": nprocs,
            "pbs_job_id": pbs_job_id,
            "firedrake_git_branch": firedrake_git_branch,
            "mesh_type": mesh_type,
            "mesh_cells_per_core": total_mesh_cells / nprocs,
            "points_per_core": points_per_core,
        }
        row.update(diagnostic_output)
        writer.writerow(row)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
