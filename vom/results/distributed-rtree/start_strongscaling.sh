#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
START_JOBS="$SCRIPT_DIR/../../start_jobs.py"

# Run from 1 to 2048 cores with 2,000,000 total random points on a
# fixed UnitCubeMesh(36, 36, 36), which has approximately 50,000 vertices.
python3 "$START_JOBS" vom_strongscaling_3d 2000000 --ncpus 64 --num_nodes 1 --mem 150 --exclusive --range
python3 "$START_JOBS" vom_strongscaling_3d 2000000 --ncpus 64 --num_nodes 2 --mem 150 --exclusive
python3 "$START_JOBS" vom_strongscaling_3d 2000000 --ncpus 64 --num_nodes 4 --mem 150 --exclusive
python3 "$START_JOBS" vom_strongscaling_3d 2000000 --ncpus 64 --num_nodes 8 --mem 200 --exclusive
python3 "$START_JOBS" vom_strongscaling_3d 2000000 --ncpus 64 --num_nodes 16 --mem 230 --exclusive
python3 "$START_JOBS" vom_strongscaling_3d 2000000 --ncpus 64 --num_nodes 32 --mem 315 --exclusive
