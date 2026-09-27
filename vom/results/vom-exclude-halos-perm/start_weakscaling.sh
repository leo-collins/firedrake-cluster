#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
START_JOBS="$SCRIPT_DIR/../../start_jobs.py"
ENV="firedrake-dev3"
EXPERIMENT=vom-exclude-halos-perm

python3 "$START_JOBS" vom_weakscaling_3d 100000 --env "$ENV" --ncpus 64 --num_nodes 1 --mem 50 --walltime 15 --exclusive --result-subdir "$EXPERIMENT" --log-subdir "$EXPERIMENT"
python3 "$START_JOBS" vom_weakscaling_3d 100000 --env "$ENV" --ncpus 64 --num_nodes 2 --mem 50 --walltime 30 --exclusive --result-subdir "$EXPERIMENT" --log-subdir "$EXPERIMENT"
python3 "$START_JOBS" vom_weakscaling_3d 100000 --env "$ENV" --ncpus 64 --num_nodes 4 --mem 50 --walltime 15 --exclusive --result-subdir "$EXPERIMENT" --log-subdir "$EXPERIMENT"
python3 "$START_JOBS" vom_weakscaling_3d 100000 --env "$ENV" --ncpus 64 --num_nodes 8 --mem 75 --walltime 30 --exclusive --result-subdir "$EXPERIMENT" --log-subdir "$EXPERIMENT"
python3 "$START_JOBS" vom_weakscaling_3d 100000 --env "$ENV" --ncpus 64 --num_nodes 16 --mem 100 --walltime 90 --exclusive --result-subdir "$EXPERIMENT" --log-subdir "$EXPERIMENT"
python3 "$START_JOBS" vom_weakscaling_3d 100000 --env "$ENV" --ncpus 64 --num_nodes 32 --mem 200 --walltime 150 --exclusive --result-subdir "$EXPERIMENT" --log-subdir "$EXPERIMENT"
