from importlib.util import find_spec
from pathlib import Path
import subprocess

import numpy as np
from firedrake.mesh import VertexOnlyMeshSF
from firedrake.petsc import garbage_cleanup


N_RUNS = 10

# Retain the candidate SF from the most recent VOM construction. Reading its
# graph after assembly keeps diagnostics out of the measured assembly path.
_candidate_sf = None
_original_candidate_sf = VertexOnlyMeshSF.candidate_sf


def _recording_candidate_sf(cls, parent_mesh, root_coordinates):
    global _candidate_sf
    _candidate_sf = _original_candidate_sf(parent_mesh, root_coordinates)
    return _candidate_sf


VertexOnlyMeshSF.candidate_sf = classmethod(_recording_candidate_sf)

_SPATIAL_INDEX_CACHE_NAMES = (
    "_bounding_box_coords_cache",
    "_rtree_cache",
    "__box_ratio_heuristic_cache",
    "_distributed_rtree_cache",
)


def _firedrake_git_commit():
    spec = find_spec("firedrake")
    if spec is None:
        return "unknown"
    if spec.origin is not None:
        repository = Path(spec.origin).resolve().parents[1]
    elif spec.submodule_search_locations:
        repository = Path(next(iter(spec.submodule_search_locations))).resolve()
    else:
        return "unknown"
    try:
        return subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def clear_spatial_index_caches(mesh):
    """Clear source-mesh data structures used to construct a VOM."""
    for name in _SPATIAL_INDEX_CACHE_NAMES:
        mesh.__dict__.pop(name, None)


def reset_cross_mesh_caches(interpolate_expr, source_mesh):
    """Reset cross-mesh construction state while retaining compiled code."""
    interpolate_expr.__dict__.pop("_interpolator", None)
    clear_spatial_index_caches(source_mesh)
    garbage_cleanup(source_mesh)


def problem_metadata(source_mesh, source_space, target_space, comm):
    firedrake_git_commit = comm.bcast(
        _firedrake_git_commit() if comm.rank == 0 else None,
        root=0,
    )
    return {
        "firedrake_git_commit": firedrake_git_commit,
        "source_cells": comm.allreduce(source_mesh.cell_set.size),
        "source_dofs": source_space.dim(),
        "target_dofs": target_space.dim(),
    }


def point_metadata(interpolate_expr, target_dofs, comm):
    point_evaluation, _ = interpolate_expr._interpolator._symbolic_expressions
    vom = point_evaluation.function_space().mesh()
    found_points = comm.allreduce(vom.cell_set.size)
    return {
        "found_target_points": found_points,
        "missing_target_points": target_dofs - found_points,
    }


def rank_diagnostic_metadata(vom, parent_mesh, local_time_s, comm):
    """Summarize the final construction's per-rank VOM diagnostics."""
    if _candidate_sf is None:
        raise RuntimeError("VOM construction did not create a candidate SF")
    boxes = parent_mesh._box_ratio_heuristic
    side_lengths = boxes[:, 1, :] - boxes[:, 0, :]
    rank_diagnostics = {
        "local_time_s": local_time_s,
        "candidate_roots": _candidate_sf.nroots,
        "candidate_leaves": _candidate_sf.nleaves,
        "candidate_input_peers": np.unique(_candidate_sf.input_ranks).size,
        "partition_boxes": boxes.shape[0],
        "partition_box_volume_sum": np.prod(side_lengths, axis=1).sum(),
        "parent_owned_cells": parent_mesh.cell_set.size,
        "parent_halo_cells": parent_mesh.cell_set.total_size - parent_mesh.cell_set.size,
        "vom_owned_points": vom.cell_set.size,
        "vom_halo_points": vom.cell_set.total_size - vom.cell_set.size,
    }
    gathered = comm.gather(rank_diagnostics, root=0)
    if comm.rank != 0:
        return {}
    output = {}
    for name in rank_diagnostics:
        values = np.asarray([row[name] for row in gathered])
        output[f"{name}_min"] = values.min()
        output[f"{name}_median"] = np.median(values)
        output[f"{name}_max"] = values.max()
    return output


def interpolation_vom(interpolate_expr):
    point_evaluation, _ = interpolate_expr._interpolator._symbolic_expressions
    return point_evaluation.function_space().mesh()
