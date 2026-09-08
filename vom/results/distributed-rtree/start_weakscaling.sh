#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
START_JOBS="$SCRIPT_DIR/../../start_jobs.py"
ENV="firedrake-dev"

# Run from 1 to 2048 cores with 100,000 random points per core on a
# mesh with approximately 25,000 tetrahedra per core. The global mesh size is
# scaled from the one-core baseline UnitCubeMesh(16, 16, 16).
python3 "$START_JOBS" vom_weakscaling_3d 100000 --env "$ENV" --ncpus 64 --num_nodes 1 --mem 150 --exclusive --range
python3 "$START_JOBS" vom_weakscaling_3d 100000 --env "$ENV" --ncpus 64 --num_nodes 2 --mem 150 --exclusive
python3 "$START_JOBS" vom_weakscaling_3d 100000 --env "$ENV" --ncpus 64 --num_nodes 4 --mem 400 --walltime 720 --exclusive
python3 "$START_JOBS" vom_weakscaling_3d 100000 --env "$ENV" --ncpus 64 --num_nodes 8 --mem 400 --walltime 720 --exclusive
python3 "$START_JOBS" vom_weakscaling_3d 100000 --env "$ENV" --ncpus 64 --num_nodes 16 --mem 400 --walltime 720 --exclusive
python3 "$START_JOBS" vom_weakscaling_3d 100000 --env "$ENV" --ncpus 64 --num_nodes 32 --mem 400 --walltime 720 --exclusive
