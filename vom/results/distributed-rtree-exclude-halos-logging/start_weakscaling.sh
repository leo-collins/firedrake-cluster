#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
START_JOBS="$SCRIPT_DIR/../../start_jobs.py"
ENV="firedrake-dev2"
EXPERIMENT=distributed-rtree-exclude-halos-logging

# Install the Firedrake variant that excludes VOM halos before submitting.
# Match the baseline logging experiment: 64–2048 ranks, 100,000 points per rank.
python3 "$START_JOBS" vom_weakscaling_3d_flamegraph 100000 --env "$ENV" --ncpus 64 --num_nodes 1 --mem 150 --exclusive --result-subdir "$EXPERIMENT" --log-subdir "$EXPERIMENT"
python3 "$START_JOBS" vom_weakscaling_3d_flamegraph 100000 --env "$ENV" --ncpus 64 --num_nodes 2 --mem 150 --exclusive --result-subdir "$EXPERIMENT" --log-subdir "$EXPERIMENT"
python3 "$START_JOBS" vom_weakscaling_3d_flamegraph 100000 --env "$ENV" --ncpus 64 --num_nodes 4 --mem 400 --walltime 720 --exclusive --result-subdir "$EXPERIMENT" --log-subdir "$EXPERIMENT"
python3 "$START_JOBS" vom_weakscaling_3d_flamegraph 100000 --env "$ENV" --ncpus 64 --num_nodes 8 --mem 400 --walltime 720 --exclusive --result-subdir "$EXPERIMENT" --log-subdir "$EXPERIMENT"
python3 "$START_JOBS" vom_weakscaling_3d_flamegraph 100000 --env "$ENV" --ncpus 64 --num_nodes 16 --mem 400 --walltime 720 --exclusive --result-subdir "$EXPERIMENT" --log-subdir "$EXPERIMENT"
python3 "$START_JOBS" vom_weakscaling_3d_flamegraph 100000 --env "$ENV" --ncpus 64 --num_nodes 32 --mem 400 --walltime 720 --exclusive --result-subdir "$EXPERIMENT" --log-subdir "$EXPERIMENT"
