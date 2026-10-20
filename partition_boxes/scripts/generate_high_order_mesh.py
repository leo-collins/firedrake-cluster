"""Generate a distributed unit cube with warped, cubic tetrahedral geometry.

Run, for example, from the repository root::

    mpiexec -n 4 python partition_boxes/scripts/generate_high_order_mesh.py

Netgen generates only a coarse seed on rank zero. Firedrake distributes it
and uniformly refines it in parallel before interpolating the warp into CG3.
The default target is 50,000 cells per MPI rank (within approximately 10%).
The checkpoint preserves the geometry and partition and can be loaded with::

    with CheckpointFile(path, "r") as checkpoint:
        mesh = checkpoint.load_mesh("high_order_cube")
"""

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI
from netgen.csg import unit_cube
from netgen.meshing import Mesh as NetgenMesh

from firedrake import (
    COMM_WORLD,
    CheckpointFile,
    DirichletBC,
    Function,
    FunctionSpace,
    Mesh,
    MeshHierarchy,
    PETSc,
    SpatialCoordinate,
    VectorFunctionSpace,
    as_vector,
    det,
    grad,
    pi,
    sin,
)


def coarse_mesh(target_cells, comm):
    """Tune Netgen's maxh on rank zero, with collective error reporting."""
    ngmesh = NetgenMesh(dim=3)
    result = None
    if comm.rank == 0:
        try:
            maxh = (6 / target_cells) ** (1 / 3)
            best = None
            for _ in range(6):
                candidate = unit_cube.GenerateMesh(maxh=maxh)
                error = abs(candidate.ne / target_cells - 1)
                if best is None or error < best[0]:
                    best = (error, candidate, maxh)
                if error <= 0.1:
                    break
                maxh *= (candidate.ne / target_cells) ** (1 / 3)
            error, ngmesh, maxh = best
            if error > 0.1:
                raise RuntimeError(
                    "Could not size the coarse mesh within 10% of the target; "
                    "try fewer --refinements."
                )
            result = (ngmesh.ne, maxh, None)
        except Exception as exc:
            result = (None, None, str(exc))
    ncells, maxh, error = comm.bcast(result, root=0)
    if error is not None:
        raise RuntimeError(f"Netgen mesh generation failed: {error}")
    PETSc.Sys.Print(f"Netgen seed: {ncells} cells, maxh={maxh:.6g}")
    return Mesh(ngmesh, comm=comm), maxh


def warp_mesh(mesh, h):
    """Curve interior facets at a wavelength of roughly four cell widths."""
    x = SpatialCoordinate(mesh)
    periods = max(1, round(1 / (4 * h)))
    amplitude = 0.2 * h
    waves = [sin(2 * pi * periods * x[i]) for i in range(3)]
    displacement = as_vector([
        amplitude * sin(pi * x[i]) * waves[(i + 1) % 3] * waves[(i + 2) % 3]
        for i in range(3)
    ])
    V = VectorFunctionSpace(mesh, "CG", 3)
    coordinates = Function(V).interpolate(x + displacement)
    # Fix all boundary nodes exactly
    DirichletBC(V, x, "on_boundary").apply(coordinates)
    return Mesh(coordinates, name="high_order_cube")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", nargs="?", type=Path,
        help="checkpoint path (default: partition_boxes/meshes/high_order_cube_<N>procs.h5)",
    )
    parser.add_argument("--cells-per-rank", type=int, default=50_000)
    parser.add_argument(
        "--refinements", type=int, default=3,
        help="parallel uniform refinements; each multiplies the cell count by eight",
    )
    args = parser.parse_args()
    if args.cells_per_rank < 1:
        parser.error("--cells-per-rank must be positive")
    if args.refinements < 0:
        parser.error("--refinements must be nonnegative")

    comm = COMM_WORLD
    target = args.cells_per_rank * comm.size
    seed_target = target / 8**args.refinements
    if seed_target < max(50, comm.size):
        parser.error("coarse seed is too small; use fewer --refinements")
    output = args.output
    if output is None:
        output = (Path(__file__).resolve().parents[1] / "meshes"
                  / f"high_order_cube_{comm.size}procs.h5")

    PETSc.Sys.Print(
        f"Generating on {comm.size} ranks: target={target} cells, "
        f"parallel refinements={args.refinements}"
    )
    seed, maxh = coarse_mesh(seed_target, comm)
    # Planar cube faces need no CAD snapping during refinement. Use the
    # linear geometry; a fresh Function avoids reusing the cached Netgen mesh.
    seed = Mesh(seed.coordinates.copy())
    mesh = MeshHierarchy(seed, args.refinements)[-1]
    mesh = warp_mesh(mesh, maxh / 2**args.refinements)
    owned = mesh.cell_set.size
    total = comm.allreduce(owned, op=MPI.SUM)
    smallest = comm.allreduce(owned, op=MPI.MIN)
    largest = comm.allreduce(owned, op=MPI.MAX)
    PETSc.Sys.Print(
        f"Generated {total} tetrahedra: {total / comm.size:.1f} cells/rank, "
        f"owned range=[{smallest}, {largest}]"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with CheckpointFile(str(output), "w", comm=comm) as checkpoint:
        checkpoint.save_mesh(mesh)
    PETSc.Sys.Print(f"Saved high_order_cube to {output}")


if __name__ == "__main__":
    main()
