from importlib.util import find_spec
from pathlib import Path
import subprocess

from firedrake.petsc import garbage_cleanup


N_RUNS = 10

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
