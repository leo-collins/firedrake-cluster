"""
VertexOnlyMesh creation benchmark.

Usage:
    mpirun -n N python vom_script.py [--nx NX] [--npoints NPOINTS] [--ntrials NTRIALS] [--output OUTPUT]

Defaults: nx=500, npoints=10_000_000, ntrials=5, output=results/vom_results.csv
"""
import argparse
import csv
import datetime
import os
import socket
from time import perf_counter_ns

import numpy as np
from firedrake import *
from firedrake.petsc import PETSc
from mpi4py import MPI


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_benchmark(mesh, points, comm):
    comm.barrier()
    t_start = perf_counter_ns()
    vom = VertexOnlyMesh(mesh, points)
    comm.barrier()
    t_end = perf_counter_ns()
    elapsed_s = (t_end - t_start) / 1e9

    PETSc.Sys.Print(f"done in {elapsed_s:.3f}s")

    return elapsed_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=500)
    parser.add_argument("--npoints", type=int, default=10_000_000)
    parser.add_argument("--output", type=str, default=os.path.join(SCRIPT_DIR, "results", "vom_results.csv"))
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    PETSc.Sys.Print(f"Starting benchmark: nprocs={nprocs}, nx={args.nx}, npoints={args.npoints}")

    PETSc.Sys.Print("Building mesh...")
    mesh = UnitSquareMesh(args.nx, args.nx)
    PETSc.Sys.Print("Mesh built.")

    # All ranks use the same seed for reproducibility
    rng = np.random.default_rng(42)
    points = rng.random((args.npoints, 2))

    PETSc.Sys.Print(f"Creating VOM")
    elapsed = run_benchmark(mesh, points, comm)

    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        row = {
            "date": datetime.datetime.now().isoformat(),
            "hostname": socket.gethostname(),
            "nprocs": nprocs,
            "nx": args.nx,
            "npoints": args.npoints,
            "time": elapsed,
        }

        write_header = not os.path.exists(args.output)
        with open(args.output, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print(
            f"nprocs={nprocs}, nx={args.nx}, npoints={args.npoints}: "
            f"time={elapsed:.3f}s, "
        )


if __name__ == "__main__":
    main()
