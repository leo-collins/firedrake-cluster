#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
START_JOBS="$SCRIPT_DIR/../../start_jobs.py"

# Run from 1 to 2048 cores with 100,000,000 total random points on a
# fixed UnitCubeMesh(162, 162, 162), which has 25,509,168 tetrahedra.
python3 "$START_JOBS" vom_strongscaling_3d 100000000 --ncpus 64 --num_nodes 1 --mem 200 --walltime 720 --exclusive --range
python3 "$START_JOBS" vom_strongscaling_3d 100000000 --ncpus 64 --num_nodes 2 --mem 150 --exclusive
python3 "$START_JOBS" vom_strongscaling_3d 100000000 --ncpus 64 --num_nodes 4 --mem 150 --exclusive
python3 "$START_JOBS" vom_strongscaling_3d 100000000 --ncpus 64 --num_nodes 8 --mem 200 --exclusive
python3 "$START_JOBS" vom_strongscaling_3d 100000000 --ncpus 64 --num_nodes 16 --mem 230 --exclusive
python3 "$START_JOBS" vom_strongscaling_3d 100000000 --ncpus 64 --num_nodes 32 --mem 315 --walltime 720 --exclusive
