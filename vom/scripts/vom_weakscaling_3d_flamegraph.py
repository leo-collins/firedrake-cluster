import csv
import fcntl
import os
import sys
from pathlib import Path
from time import perf_counter_ns
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from mpi4py import MPI


# Run with:
#   mpiexec -n <nranks> python vom_weakscaling_3d_flamegraph.py \
#       <points_per_rank> [csv_path] [pbs_job_id]
# VOM_FLAMEGRAPH_SUBDIR selects the experiment directory below vom/flamegraphs.
# An explicit -log_view overrides the default flamegraph destination.
if len(sys.argv) < 2:
    raise ValueError(
        "Usage: vom_weakscaling_3d_flamegraph.py "
        "<points_per_rank> [csv_path] [pbs_job_id]"
    )

comm = MPI.COMM_WORLD
nprocs = comm.size
points_per_rank = int(sys.argv[1])
csv_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
pbs_job_id = sys.argv[3] if len(sys.argv) > 3 else None
if points_per_rank < 1:
    raise ValueError("points_per_rank must be >= 1")

flamegraph_subdir = Path(
    os.environ.get("VOM_FLAMEGRAPH_SUBDIR", "distributed-rtree-logging")
)
if flamegraph_subdir.is_absolute() or ".." in flamegraph_subdir.parts:
    raise ValueError(
        "VOM_FLAMEGRAPH_SUBDIR must stay below the flamegraphs directory"
    )
flamegraph_dir = (
    Path(__file__).resolve().parents[1]
    / "flamegraphs"
    / flamegraph_subdir
)
if comm.rank == 0:
    flamegraph_dir.mkdir(parents=True, exist_ok=True)
comm.Barrier()
if "-log_view" not in sys.argv:
    flamegraph_path = flamegraph_dir / f"vom_flamegraph_{nprocs}.txt"
    sys.argv.extend(["-log_view", f":{flamegraph_path}:ascii_flamegraph"])

from firedrake import PETSc, UnitCubeMesh, VertexOnlyMesh  # noqa: E402
from firedrake.mesh import VertexOnlyMeshSF  # noqa: E402
from firedrake.petsc import garbage_cleanup  # noqa: E402
from firedrake.utils import IntType  # noqa: E402


def clear_spatial_index_caches(mesh):
    for cache in (
        "_partition_rtree_cache",
        "_distributed_rtree_cache",
        "_rtree_cache",
        "__box_ratio_heuristic_cache",
        "_box_ratio_heuristic_cache",  # Compatibility with older Firedrake.
        "_bounding_box_coords_cache",
    ):
        setattr(mesh, cache, None)


def range_text(values, digits=1):
    values = np.asarray(values)
    return f"{values.min():.0f}/{np.median(values):.{digits}f}/{values.max():.0f}"


def add_summary(row, name, values):
    values = np.asarray(values)
    row[f"{name}_min"] = values.min()
    row[f"{name}_median"] = np.median(values)
    row[f"{name}_max"] = values.max()


def histogram_summary(histogram):
    populated = np.flatnonzero(histogram)
    cumulative = np.cumsum(histogram)
    count = int(cumulative[-1])
    lower = np.searchsorted(cumulative, (count - 1) // 2, side="right")
    upper = np.searchsorted(cumulative, count // 2, side="right")
    return int(populated[0]), (lower + upper) / 2, int(populated[-1])


# Retain the candidate SF created by VertexOnlyMesh without adding diagnostics
# to the timed construction path.
candidate_observation = None
original_candidate_sf = VertexOnlyMeshSF.candidate_sf


def recording_candidate_sf(cls, parent_mesh, root_coordinates):
    global candidate_observation
    candidate_observation = original_candidate_sf(parent_mesh, root_coordinates)
    return candidate_observation


VertexOnlyMeshSF.candidate_sf = classmethod(recording_candidate_sf)

# Keep approximately 25,000 tetrahedra per rank.
target_mesh_cells_per_rank = 25_000
mesh_n = max(round((target_mesh_cells_per_rank * nprocs / 6) ** (1 / 3)), 1)
total_mesh_cells = 6 * mesh_n**3
PETSc.Sys.Print(
    f"nprocs={nprocs}: UnitCubeMesh({mesh_n}, {mesh_n}, {mesh_n}), "
    f"cells/rank={total_mesh_cells / nprocs:.1f}, "
    f"points/rank={points_per_rank}"
)

mesh = UnitCubeMesh(mesh_n, mesh_n, mesh_n)
mesh.tolerance = 0.5

bit_generator = np.random.PCG64(0)
bit_generator.advance(3 * comm.rank * points_per_rank)
points = np.random.Generator(bit_generator).random((points_per_rank, 3))

# Warm generated code and one-time Firedrake setup outside the measured runs.
with PETSc.Log.Event("create_vom_warmup"):
    vom = VertexOnlyMesh(mesh, points, redundant=False)

times = []
last_candidate_sf = None
final_rank_diagnostics = None
for run in range(10):
    del vom
    clear_spatial_index_caches(mesh)
    garbage_cleanup(mesh)
    comm.Barrier()

    candidate_observation = None
    with PETSc.Log.Event(f"create_vom_{run:02d}"):
        start = perf_counter_ns()
        vom = VertexOnlyMesh(mesh, points, redundant=False)
        local_time = (perf_counter_ns() - start) / 1e9

    if candidate_observation is None:
        raise RuntimeError("VertexOnlyMesh did not construct a candidate SF")
    last_candidate_sf = candidate_observation

    boxes = mesh._box_ratio_heuristic
    side_lengths = boxes[:, 1, :] - boxes[:, 0, :]
    rank_diagnostics = {
        "local_time_s": local_time,
        "candidate_roots": last_candidate_sf.nroots,
        "candidate_leaves": last_candidate_sf.nleaves,
        "candidate_input_peers": np.unique(last_candidate_sf.input_ranks).size,
        "partition_boxes": boxes.shape[0],
        "partition_box_volume_sum": np.prod(side_lengths, axis=1).sum(),
        "parent_owned_cells": mesh.cell_set.size,
        "parent_halo_cells": mesh.cell_set.total_size - mesh.cell_set.size,
        "vom_owned_points": vom.cell_set.size,
        "vom_halo_points": vom.cell_set.total_size - vom.cell_set.size,
    }
    gathered = comm.gather(rank_diagnostics, root=0)
    max_time = comm.allreduce(local_time, op=MPI.MAX)
    times.append(max_time)

    if comm.rank == 0:
        if run == 9:
            final_rank_diagnostics = gathered
        candidates = [row["candidate_leaves"] for row in gathered]
        boxes_per_rank = [row["partition_boxes"] for row in gathered]
        peers = [row["candidate_input_peers"] for row in gathered]
        PETSc.Sys.Print(
            f"run {run:02d}: max={max_time:.6f}s; "
            f"candidates[min/median/max]={range_text(candidates)}, "
            f"boxes[min/median/max]={range_text(boxes_per_rank)}, "
            f"total_boxes={sum(boxes_per_rank)}, "
            f"input_peers[min/median/max]={range_text(peers)}"
        )

# Compute the exact number of candidate leaves attached to each input root once.
# This is outside every create_vom event so it does not affect their timings.
with PETSc.Log.Event("collect_candidate_fanout_diagnostics"):
    leaf_ones = np.ones(last_candidate_sf.leaf_buffer_size, dtype=IntType)
    root_fanout = np.zeros(last_candidate_sf.nroots, dtype=IntType)
    last_candidate_sf.reduce(leaf_ones, root_fanout, op=MPI.SUM)
local_histogram = np.bincount(root_fanout, minlength=nprocs + 1)
global_histogram = np.zeros_like(local_histogram) if comm.rank == 0 else None
comm.Reduce(local_histogram, global_histogram, op=MPI.SUM, root=0)

if comm.rank == 0:
    if csv_path is None:
        diagnostics_path = flamegraph_dir / "vom_weakscaling_diagnostics.csv"
    else:
        diagnostics_path = csv_path

    output = {
        "nprocs": nprocs,
        "pbs_job_id": pbs_job_id,
        "mesh_n": mesh_n,
        "total_mesh_cells": total_mesh_cells,
        "mesh_cells_per_rank": total_mesh_cells / nprocs,
        "points_per_rank": points_per_rank,
        "total_points": points_per_rank * nprocs,
    }
    add_summary(output, "vom_time_s", times)
    for name in final_rank_diagnostics[0]:
        add_summary(output, name, [row[name] for row in final_rank_diagnostics])
    root_min, root_median, root_max = histogram_summary(global_histogram)
    output["root_fanout_min"] = root_min
    output["root_fanout_median"] = root_median
    output["root_fanout_max"] = root_max

    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    # The node-count jobs are submitted separately and can finish concurrently.
    # Lock the shared CSV while appending its one row for this MPI size.
    with diagnostics_path.open("a+", newline="") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0, 2)
        writer = csv.DictWriter(stream, fieldnames=output.keys())
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(output)
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)

    PETSc.Sys.Print(
        f"final-root-fanout[min/median/max]="
        f"{root_min}/{root_median:.1f}/{root_max}"
    )
    PETSc.Sys.Print(f"Diagnostics: {diagnostics_path}")
    PETSc.Sys.Print(
        f"VOM time[min/median/max]={min(times):.6f}/"
        f"{np.median(times):.6f}/{max(times):.6f}s"
    )

del vom
garbage_cleanup(mesh)
