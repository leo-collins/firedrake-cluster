#!/bin/bash

mpirun -n 1 python ../../vom_simple.py -log_view :vom_simple1.txt:ascii_flamegraph
mpirun -n 2 python ../../vom_simple.py -log_view :vom_simple2.txt:ascii_flamegraph
mpirun -n 4 python ../../vom_simple.py -log_view :vom_simple4.txt:ascii_flamegraph
mpirun -n 8 python ../../vom_simple.py -log_view :vom_simple8.txt:ascii_flamegraph
mpirun -n 16 python ../../vom_simple.py -log_view :vom_simple16.txt:ascii_flamegraph
mpirun -n 32 python ../../vom_simple.py -log_view :vom_simple32.txt:ascii_flamegraph
mpirun -n 64 python ../../vom_simple.py -log_view :vom_simple64.txt:ascii_flamegraph
mpirun -n 128 python ../../vom_simple.py -log_view :vom_simple128.txt:ascii_flamegraph
