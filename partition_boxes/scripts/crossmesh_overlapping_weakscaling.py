import csv
import gc
from math import floor, ceil
from pathlib import Path
from sys import argv, path
from time import perf_counter_ns

from benchmark_utils import (
    N_RUNS, interpolation_vom, point_metadata, problem_metadata,
    rank_diagnostic_metadata, reset_cross_mesh_caches,
)
from firedrake import (
    COMM_WORLD, PETSc, Function, FunctionSpace, TrialFunction, UnitCubeMesh,
    assemble, interpolate,
)
from firedrake.petsc import garbage_cleanup
from mpi4py import MPI

path.insert(0, str(Path(__file__).resolve().parents[1] / "meshes"))
from generate_cube_hierarchy import hierarchy_sizes, refined_cube

# Weak scaling of cross-mesh interpolation with fully overlapping meshes.
# Benchmark both direct cubes and independently refined cube hierarchies.
# Run with:
#   mpirun -n <nprocs> python crossmesh_overlapping_weakscaling.py <dofs_per_core> <degree> [csv_path] [pbs_job_id]


def measure_interpolation(expr, source_mesh, label, **assembly_options):
    # Compile once, then rebuild the interpolator and spatial indexes each run.
    result = assemble(expr, **assembly_options)
    del result
    run_times = []
    for run_idx in range(N_RUNS):
        reset_cross_mesh_caches(expr, source_mesh)
        COMM_WORLD.barrier()
        t0 = perf_counter_ns()
        result = assemble(expr, **assembly_options)
        elapsed = perf_counter_ns() - t0
        run_time = COMM_WORLD.allreduce(elapsed, op=MPI.MAX) / 1e9
        PETSc.Sys.Print(f"{label}_run{run_idx} time={run_time:.6g}s")
        run_times.append(run_time)
        del result
    return run_times


def run_benchmark(n, target_n, degree, mesh_method, csv_path, pbs_jobid):
    n_cores = COMM_WORLD.size
    label = f"mesh={mesh_method}, nprocs={n_cores}"
    COMM_WORLD.barrier()
    t0_mesh = perf_counter_ns()
    if mesh_method == "UnitCubeMesh":
        mesh1 = UnitCubeMesh(n, n, n)
        mesh2 = UnitCubeMesh(target_n, target_n, target_n)
    else:
        source_seed, target_seed, levels = hierarchy_sizes(n, target_n, degree)
        PETSc.Sys.Print(
            f"nprocs={n_cores}: MeshHierarchy seeds={source_seed}/{target_seed}, refinements={levels}"
        )
        mesh1 = refined_cube(source_seed, levels)
        mesh2 = refined_cube(target_seed, levels)
    t1_mesh = perf_counter_ns()
    mesh_gen_time_s = COMM_WORLD.allreduce(t1_mesh - t0_mesh, op=MPI.MAX) / 1e9
    PETSc.Sys.Print(f"{label}: mesh setup={mesh_gen_time_s:.6g}s")

    V = FunctionSpace(mesh1, "CG", degree)
    W = FunctionSpace(mesh2, "CG", degree)
    metadata = problem_metadata(mesh1, V, W, COMM_WORLD)

    interp = interpolate(TrialFunction(V), W)

    twoform_run_times_s = measure_interpolation(
        interp, mesh1, f"{label}: cold-index twoform", mat_type="aij",
    )
    metadata.update(point_metadata(interp, W.dim(), COMM_WORLD))
    metadata.update(rank_diagnostic_metadata(interpolation_vom(interp), mesh1, COMM_WORLD))
    reset_cross_mesh_caches(interp, mesh1)

    # A Function operand evaluates source values and broadcasts them through the
    # VOM SF, without assembling an AIJ interpolation or permutation matrix.
    source_function = Function(V).assign(1)
    oneform_interp = interpolate(source_function, W)
    oneform_run_times_s = measure_interpolation(
        oneform_interp, mesh1, f"{label}: cold-index oneform",
    )
    reset_cross_mesh_caches(oneform_interp, mesh1)
    average_dofs_per_core = (W.dim() + V.dim()) / (2 * n_cores)

    if COMM_WORLD.rank != 0 or csv_path is None:
        return

    row = {
        "mesh": mesh_method,
        "nprocs": n_cores,
        "pbs_job_id": pbs_jobid,
        "degree": degree,
        "dofs_per_core": average_dofs_per_core,
        "mesh_gen_time_s": mesh_gen_time_s,
        **metadata,
        **{f"twoform_run{i}": value for i, value in enumerate(twoform_run_times_s)},
        **{f"oneform_run{i}": value for i, value in enumerate(oneform_run_times_s)},
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if not write_header:
        with csv_path.open(newline="") as f:
            if next(csv.reader(f)) != list(row):
                raise ValueError(f"CSV header does not match the benchmark columns: {csv_path}. Use a new CSV path or move the existing file before running.")
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    if len(argv) < 3:
        raise ValueError("Usage: crossmesh_overlapping_weakscaling.py <dofs_per_core> <degree> [csv_path] [pbs_job_id]")
    dofs_per_core = int(argv[1])
    if dofs_per_core < 1:
        raise ValueError("dofs_per_core must be >= 1")
    degree = int(argv[2])
    if degree < 1:
        raise ValueError("degree must be >= 1")
    csv_path = Path(argv[3]) if len(argv) > 3 else None
    pbs_jobid = argv[4] if len(argv) > 4 else None

    # For UnitCubeMesh, dim(CG(degree)) = (degree * n + 1)^3.
    n = max(floor((((dofs_per_core * COMM_WORLD.size) ** (1 / 3)) - 1) / degree), 1)
    target_n = ceil(1.01 * n)
    for mesh_method in ("UnitCubeMesh", "MeshHierarchy"):
        run_benchmark(n, target_n, degree, mesh_method, csv_path, pbs_jobid)
        gc.collect()
        garbage_cleanup(COMM_WORLD)


if __name__ == "__main__":
    main()
