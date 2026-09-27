#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
START_JOBS="$SCRIPT_DIR/../../start_jobs.py"
ENV="firedrake-dev3"
EXPERIMENT=vom-exclude-halos-perm

# Weak-scale the VOM-only test from 64 to 2048 ranks, with 100,000 points
# per rank and approximately 25,000 tetrahedra per rank.
# Walltimes allow for mesh setup and MPI startup as well as the measured runs.
# The much longer 16-node log was stalled after starting alongside other sizes;
# its measured runs project to under an hour. 2048 ranks has no successful log.
# Memory requests are per node; the 2048-rank job keeps the maximum after an
# earlier weakscaling run OOMed with 315 GB requested per node.
python3 "$START_JOBS" vom_weakscaling_3d 100000 --env "$ENV" --ncpus 64 --num_nodes 1 --mem 50 --walltime 15 --exclusive --result-subdir "$EXPERIMENT" --log-subdir "$EXPERIMENT"
python3 "$START_JOBS" vom_weakscaling_3d 100000 --env "$ENV" --ncpus 64 --num_nodes 2 --mem 50 --walltime 30 --exclusive --result-subdir "$EXPERIMENT" --log-subdir "$EXPERIMENT"
python3 "$START_JOBS" vom_weakscaling_3d 100000 --env "$ENV" --ncpus 64 --num_nodes 4 --mem 50 --walltime 15 --exclusive --result-subdir "$EXPERIMENT" --log-subdir "$EXPERIMENT"
python3 "$START_JOBS" vom_weakscaling_3d 100000 --env "$ENV" --ncpus 64 --num_nodes 8 --mem 75 --walltime 30 --exclusive --result-subdir "$EXPERIMENT" --log-subdir "$EXPERIMENT"
python3 "$START_JOBS" vom_weakscaling_3d 100000 --env "$ENV" --ncpus 64 --num_nodes 16 --mem 100 --walltime 60 --exclusive --result-subdir "$EXPERIMENT" --log-subdir "$EXPERIMENT"
python3 "$START_JOBS" vom_weakscaling_3d 100000 --env "$ENV" --ncpus 64 --num_nodes 32 --mem 400 --walltime 60 --exclusive --result-subdir "$EXPERIMENT" --log-subdir "$EXPERIMENT"
