#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
START_JOBS="$SCRIPT_DIR/../../start_jobs.py"

# Use all 64 cores on each node and weak-scale both the mesh and the random
# points.  Each job writes vom_flamegraph_<nranks>.txt in this directory and
# appends one min/median/max summary row to the shared results CSV.
python3 "$START_JOBS" vom_weakscaling_3d_flamegraph 100000 --ncpus 64 --num_nodes 1 --mem 150 --exclusive --result-subdir distributed-rtree-logging
python3 "$START_JOBS" vom_weakscaling_3d_flamegraph 100000 --ncpus 64 --num_nodes 2 --mem 150 --exclusive --result-subdir distributed-rtree-logging
python3 "$START_JOBS" vom_weakscaling_3d_flamegraph 100000 --ncpus 64 --num_nodes 4 --mem 400 --walltime 720 --exclusive --result-subdir distributed-rtree-logging
python3 "$START_JOBS" vom_weakscaling_3d_flamegraph 100000 --ncpus 64 --num_nodes 8 --mem 400 --walltime 720 --exclusive --result-subdir distributed-rtree-logging
python3 "$START_JOBS" vom_weakscaling_3d_flamegraph 100000 --ncpus 64 --num_nodes 16 --mem 400 --walltime 720 --exclusive --result-subdir distributed-rtree-logging
python3 "$START_JOBS" vom_weakscaling_3d_flamegraph 100000 --ncpus 64 --num_nodes 32 --mem 400 --walltime 720 --exclusive --result-subdir distributed-rtree-logging
