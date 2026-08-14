#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this launcher, so it can be run from any directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
START_JOBS="$SCRIPT_DIR/../../start_jobs.py"
ENV="firedrake-dev3"

# Run contacting strong scaling from 1 through 2048 MPI ranks with
# 10,000,000 nominal total dofs and CG3 elements.

python3 "$START_JOBS" contacting_strongscaling_3d 10000000 3 --env "$ENV" --result-subdir distributed-rtree-perm --ncpus 64 --num_nodes 1 --mem 400 --exclusive --range
python3 "$START_JOBS" contacting_strongscaling_3d 10000000 3 --env "$ENV" --result-subdir distributed-rtree-perm --ncpus 64 --num_nodes 2 --mem 400 --exclusive
python3 "$START_JOBS" contacting_strongscaling_3d 10000000 3 --env "$ENV" --result-subdir distributed-rtree-perm --ncpus 64 --num_nodes 4 --mem 400 --exclusive
python3 "$START_JOBS" contacting_strongscaling_3d 10000000 3 --env "$ENV" --result-subdir distributed-rtree-perm --ncpus 64 --num_nodes 8 --mem 400 --exclusive
python3 "$START_JOBS" contacting_strongscaling_3d 10000000 3 --env "$ENV" --result-subdir distributed-rtree-perm --ncpus 64 --num_nodes 16 --mem 400 --exclusive
python3 "$START_JOBS" contacting_strongscaling_3d 10000000 3 --env "$ENV" --result-subdir distributed-rtree-perm --ncpus 64 --num_nodes 32 --mem 400 --exclusive
