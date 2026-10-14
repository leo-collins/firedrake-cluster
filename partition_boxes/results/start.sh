#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
START_JOBS="$SCRIPT_DIR/../start_jobs.py"
ENV="firedrake-dev-int64"
BENCHMARK="crossmesh_overlapping_weakscaling"

# Both mesh approaches and all rank counts append to the same CSV in results/.
# Run 1-64 ranks sequentially, then separate jobs for 128-2048 ranks on HX1.
#python3 "$START_JOBS" "$BENCHMARK" 200000 3 --env "$ENV" --result-subdir . --ncpus 64 --num_nodes 1 --mem 100 --walltime 60 --exclusive --range
#python3 "$START_JOBS" "$BENCHMARK" 200000 3 --env "$ENV" --result-subdir . --ncpus 64 --num_nodes 2 --mem 100 --exclusive
#python3 "$START_JOBS" "$BENCHMARK" 200000 3 --env "$ENV" --result-subdir . --ncpus 64 --num_nodes 4 --mem 150 --exclusive
python3 "$START_JOBS" "$BENCHMARK" 200000 3 --env "$ENV" --result-subdir . --ncpus 64 --num_nodes 8 --mem 250 --exclusive
#python3 "$START_JOBS" "$BENCHMARK" 200000 3 --env "$ENV" --result-subdir . --ncpus 64 --num_nodes 16 --mem 300 --exclusive
#python3 "$START_JOBS" "$BENCHMARK" 200000 3 --env "$ENV" --result-subdir . --ncpus 64 --num_nodes 32 --mem 400 --exclusive
