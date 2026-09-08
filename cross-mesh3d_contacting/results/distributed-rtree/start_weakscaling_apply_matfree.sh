#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this launcher, so it can be run from any directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
START_JOBS="$SCRIPT_DIR/../../start_jobs.py"
ENV="firedrake-dev2"

# Run contacting matrix-free application weak scaling from 1 through 2048
# MPI ranks with 200,000 nominal dofs per core and CG3 elements.

python3 "$START_JOBS" contacting_apply_matfree_3d 200000 3 --env "$ENV" --result-subdir distributed-rtree --ncpus 64 --num_nodes 1 --mem 20 --walltime 10 --exclusive --range
python3 "$START_JOBS" contacting_apply_matfree_3d 200000 3 --env "$ENV" --result-subdir distributed-rtree --ncpus 64 --num_nodes 2 --mem 35 --walltime 10 --exclusive
python3 "$START_JOBS" contacting_apply_matfree_3d 200000 3 --env "$ENV" --result-subdir distributed-rtree --ncpus 64 --num_nodes 4 --mem 60 --walltime 15 --exclusive
python3 "$START_JOBS" contacting_apply_matfree_3d 200000 3 --env "$ENV" --result-subdir distributed-rtree --ncpus 64 --num_nodes 8 --mem 95 --walltime 25 --exclusive
python3 "$START_JOBS" contacting_apply_matfree_3d 200000 3 --env "$ENV" --result-subdir distributed-rtree --ncpus 64 --num_nodes 16 --mem 170 --walltime 60 --exclusive
python3 "$START_JOBS" contacting_apply_matfree_3d 200000 3 --env "$ENV" --result-subdir distributed-rtree --ncpus 64 --num_nodes 32 --mem 310 --walltime 220 --exclusive
