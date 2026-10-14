"""Save two independently refined cubes for a target rank count.

Usage: mpirun -n 256 python generate_cube_hierarchy.py 200000 3 2304
Target ranks default to the MPI size. Use up to three uniform refinements,
keeping each predicted DOF count within 3% of the original benchmark mesh.
"""

import argparse
from functools import partial
import gc
from math import ceil, floor
from pathlib import Path

from firedrake import COMM_WORLD, CheckpointFile, Mesh, MeshHierarchy, PETSc, UnitCubeMesh
from firedrake.petsc import garbage_cleanup


def hierarchy_sizes(n, m, degree):
    levels = 3
    while True:
        scale = 2 ** levels
        source_seed = max((n + scale // 2) // scale, 1)
        target_seed = max((m + scale // 2) // scale, 1)
        source_n = source_seed * scale
        target_n = target_seed * scale
        source_error = abs(((degree * source_n + 1) / (degree * n + 1)) ** 3 - 1)
        target_error = abs(((degree * target_n + 1) / (degree * m + 1)) ** 3 - 1)
        if source_n < target_n and max(source_error, target_error) <= 0.03:
            return source_seed, target_seed, levels
        levels -= 1


def refined_cube(seed_n, levels, name):
    seed = UnitCubeMesh(seed_n, seed_n, seed_n, name=name)
    hierarchy = MeshHierarchy(seed, levels, mesh_builder=partial(Mesh, name=name))
    return hierarchy[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dofs_per_rank", type=int)
    parser.add_argument("degree", type=int)
    parser.add_argument("target_ranks", type=int, nargs="?", default=COMM_WORLD.size,
                        help="Rank count used to size the meshes (defaults to MPI size).")
    args = parser.parse_args()
    if args.dofs_per_rank < 1 or args.degree < 1 or args.target_ranks < 1:
        parser.error("DOFs per rank, degree and target ranks must all be positive")

    ranks = args.target_ranks
    n = max(floor(((args.dofs_per_rank * ranks) ** (1 / 3) - 1) / args.degree), 1)
    m = ceil(1.01 * n)
    source_seed, target_seed, levels = hierarchy_sizes(n, m, args.degree)
    source_n = source_seed * 2 ** levels
    target_n = target_seed * 2 ** levels
    filename = Path(__file__).resolve().parent / (
        f"{ranks}ranks_{args.dofs_per_rank}dofs_cg{args.degree}_on{COMM_WORLD.size}_hierarchy.h5"
    )

    PETSc.Sys.Print(f"Requested source/target n={n}/{m}; achieved n={source_n}/{target_n}")
    PETSc.Sys.Print(f"Seed n={source_seed}/{target_seed}; uniform refinements={levels}")
    PETSc.Sys.Print(
        f"Predicted source/target DOFs per target rank: "
        f"{(args.degree * source_n + 1) ** 3 / ranks:.1f}/"
        f"{(args.degree * target_n + 1) ** 3 / ranks:.1f}"
    )
    # Save independently so both hierarchies need not reside in memory together.
    with CheckpointFile(str(filename), "w") as checkpoint:
        source = refined_cube(source_seed, levels, "source")
        checkpoint.save_mesh(source)
        del source
        gc.collect()
        garbage_cleanup(COMM_WORLD)

        target = refined_cube(target_seed, levels, "target")
        checkpoint.save_mesh(target)
    PETSc.Sys.Print(f"Saved {filename}; sized for {ranks} ranks, generated on {COMM_WORLD.size}")


if __name__ == "__main__":
    main()
