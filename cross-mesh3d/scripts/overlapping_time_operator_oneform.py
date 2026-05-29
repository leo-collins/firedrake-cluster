import csv
from math import floor, ceil
from pathlib import Path
from sys import argv
from time import perf_counter_ns
import warnings
warnings.filterwarnings("ignore")

from firedrake import *
from firedrake.utility_meshes import _mark_mesh_boundaries
from mpi4py import MPI

# This times the pieces of fully-overlapping cross-mesh interpolation applied
# to a one-form/function.
# Run with:
#   mpiexec -n <nprocs> python overlapping_time_operator_oneform.py <dofs_per_core> <degree> [csv_path]

if len(argv) < 3:
    raise ValueError("Usage: overlapping_time_operator_oneform.py <dofs_per_core> <degree> [csv_path]")

n_cores = COMM_WORLD.size
dofs_per_core = int(argv[1])
degree = int(argv[2])
if degree < 1:
    raise ValueError("degree must be >= 1")
csv_path = Path(argv[3]) if len(argv) > 3 else None
pbs_jobid = argv[4] if len(argv) > 4 else None

# For UnitCubeMesh, dim(CG(degree)) = (degree * n + 1)^3.
n = max(floor((((dofs_per_core * n_cores) ** (1 / 3)) - 1) / degree), 1)

# Meshes have different number of nodes to force different parallel partitions.
t0_mesh = perf_counter_ns()
mesh1 = UnitCubeMesh(n, n, n)
mesh2 = UnitCubeMesh(ceil(1.01 * n), ceil(1.01 * n), ceil(1.01 * n))
t1_mesh = perf_counter_ns()
mesh_gen_time_s = (t1_mesh - t0_mesh) / 1e9
PETSc.Sys.Print(f"nprocs={n_cores}: mesh generation={mesh_gen_time_s:.6g}s")

V1 = FunctionSpace(mesh1, "CG", degree)
V2 = FunctionSpace(mesh2, "CG", degree)


def run(V1, V2):
    f = Function(V1).assign(1.1)
    # Omega_v
    V2_element = V2.ufl_element()
    x_i = assemble(interpolate(mesh2.coordinates, VectorFunctionSpace(mesh2, V2_element))).dat.data_ro.reshape(-1, mesh2.geometric_dimension)
    t0 = perf_counter_ns()
    Omega_v = VertexOnlyMesh(mesh1, x_i, redundant=False)
    t1 = perf_counter_ns()
    Omega_v_time_s = COMM_WORLD.allreduce(t1 - t0, op=MPI.MAX) / 1e9

    t0 = perf_counter_ns()
    Omega_v_io = Omega_v.input_ordering
    t1 = perf_counter_ns()
    Omega_v_io_time_s = COMM_WORLD.allreduce(t1 - t0, op=MPI.MAX) / 1e9

    P0DG_Omega_v = FunctionSpace(Omega_v, "DG", 0)

    A_interp = interpolate(f, P0DG_Omega_v)
    t0 = perf_counter_ns()
    A_f = assemble(A_interp)
    t1 = perf_counter_ns()
    A_time_s = COMM_WORLD.allreduce(t1 - t0, op=MPI.MAX) / 1e9

    P0DG_Omega_v_io = FunctionSpace(Omega_v_io, "DG", 0)

    B_interp = interpolate(A_f, P0DG_Omega_v_io)
    t0 = perf_counter_ns()
    B_f = assemble(B_interp)
    t1 = perf_counter_ns()
    B_time_s = COMM_WORLD.allreduce(t1 - t0, op=MPI.MAX) / 1e9

    C_f = Function(V2)
    t0 = perf_counter_ns()
    C_f.dat.data_wo[:] = B_f.dat.data_ro[:]
    t1 = perf_counter_ns()
    C_time_s = COMM_WORLD.allreduce(t1 - t0, op=MPI.MAX) / 1e9


    I = interpolate(f, V2)
    t0 = perf_counter_ns()
    I_f = assemble(I)
    t1 = perf_counter_ns()
    I_time_s = COMM_WORLD.allreduce(t1 - t0, op=MPI.MAX) / 1e9

    return Omega_v_time_s, Omega_v_io_time_s, A_time_s, B_time_s, C_time_s, I_time_s


TIMING_NAMES = [
    "Omega_v_time_s",
    "Omega_v_io_time_s",
    "A_time_s",
    "B_time_s",
    "C_time_s",
    "I_time_s",
]
N_RUNS = 10

# Warmup run.
run(V1, V2)

run_times_s = []
for _ in range(N_RUNS):
    COMM_WORLD.barrier()
    run_times_s.append(run(V1, V2))

average_dofs_per_core = (V2.dim() + V1.dim()) / (2 * n_cores)

if COMM_WORLD.rank == 0:
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="") as csv_file:
            w = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "nprocs",
                    "pbs_job_id",
                    "degree",
                    "dofs_per_core",
                    "mesh_gen_time_s",
                ]
                + [
                    f"{timing_name}_run{i}"
                    for timing_name in TIMING_NAMES
                    for i in range(N_RUNS)
                ],
            )
            if write_header:
                w.writeheader()
            row = {
                "nprocs": n_cores,
                "pbs_job_id": pbs_jobid,
                "degree": degree,
                "dofs_per_core": average_dofs_per_core,
                "mesh_gen_time_s": mesh_gen_time_s,
            }
            for i, run_time_s in enumerate(run_times_s):
                for timing_name, timing_s in zip(TIMING_NAMES, run_time_s):
                    row[f"{timing_name}_run{i}"] = timing_s
            w.writerow(row)
