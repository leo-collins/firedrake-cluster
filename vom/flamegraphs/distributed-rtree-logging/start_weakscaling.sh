#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
START_JOBS="$SCRIPT_DIR/../../start_jobs.py"
ENV="firedrake-dev"

# Use all 64 cores on each node and weak-scale both the mesh and the random
# points.  Each job writes vom_flamegraph_<nranks>.txt in this directory and
# appends one min/median/max summary row to the shared results CSV.
python3 "$START_JOBS" vom_weakscaling_3d_flamegraph 100000 --env "$ENV" --ncpus 64 --num_nodes 1 --mem 150 --exclusive --result-subdir distributed-rtree-logging
python3 "$START_JOBS" vom_weakscaling_3d_flamegraph 100000 --env "$ENV" --ncpus 64 --num_nodes 2 --mem 30 --walltime 10 --exclusive --result-subdir distributed-rtree-logging
python3 "$START_JOBS" vom_weakscaling_3d_flamegraph 100000 --env "$ENV" --ncpus 64 --num_nodes 4 --mem 35 --walltime 10 --exclusive --result-subdir distributed-rtree-logging
python3 "$START_JOBS" vom_weakscaling_3d_flamegraph 100000 --env "$ENV" --ncpus 64 --num_nodes 8 --mem 50 --walltime 15 --exclusive --result-subdir distributed-rtree-logging
python3 "$START_JOBS" vom_weakscaling_3d_flamegraph 100000 --env "$ENV" --ncpus 64 --num_nodes 16 --mem 400 --walltime 720 --exclusive --result-subdir distributed-rtree-logging
python3 "$START_JOBS" vom_weakscaling_3d_flamegraph 100000 --env "$ENV" --ncpus 64 --num_nodes 32 --mem 400 --walltime 720 --exclusive --result-subdir distributed-rtree-logging
