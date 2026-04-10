import numpy as np
from firedrake import *
from time import perf_counter_ns
from mpi4py import MPI
import warnings
warnings.filterwarnings("ignore")


PETSc.Sys.Print(f"nranks={COMM_WORLD.size}")

mesh = UnitSquareMesh(100, 100)
PETSc.Sys.Print(f"Finished creating mesh")

rng = np.random.default_rng(COMM_WORLD.rank)
points = rng.random((100_000, 2))

t0 = perf_counter_ns()
vom = VertexOnlyMesh(mesh, points, redundant=False)
t1 = perf_counter_ns()

t = COMM_WORLD.allreduce(t1 - t0, op=MPI.SUM) / COMM_WORLD.size
PETSc.Sys.Print(f"Time to create VertexOnlyMesh: {t / 1e9} s")
