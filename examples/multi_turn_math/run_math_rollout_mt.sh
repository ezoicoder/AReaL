#!/bin/bash
set -euo pipefail

TRIAL_NAME=math_rollout_$(date +%Y%m%d_%H%M%S)
LOG_FILE="${TRIAL_NAME}.log"

echo "Starting trial: ${TRIAL_NAME}"
echo "Logging to: ${LOG_FILE}"

script -f -c "python3 -m areal.launcher.local examples/rollout_only/rollout.py \
  --config examples/multi_turn_math/math_rollout_mt.yaml \
  trial_name=${TRIAL_NAME}" "${LOG_FILE}"
