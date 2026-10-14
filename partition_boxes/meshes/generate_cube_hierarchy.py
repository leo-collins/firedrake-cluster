"""Build refined cubes with DOF counts within 3% of the benchmark meshes."""

from firedrake import MeshHierarchy, UnitCubeMesh


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


def refined_cube(seed_n, levels):
    seed = UnitCubeMesh(seed_n, seed_n, seed_n)
    hierarchy = MeshHierarchy(seed, levels)
    return hierarchy[-1]
