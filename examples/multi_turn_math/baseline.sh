#!/bin/bash
set -e  # Exit on error

# Define trial name with timestamp
TRIAL_NAME="baseline_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${TRIAL_NAME}.log"

echo "Starting trial: ${TRIAL_NAME}"
echo "Logging to: ${LOG_FILE}"

script -f -c "python3 -m areal.launcher.local examples/multi_turn_math/gsm8k_rl_mt.py --config examples/multi_turn_math/gsm8k_grpo_attn.yaml \
        experiment_name=math trial_name=${TRIAL_NAME} \
        +actor.pad_to_maximum=True \
        +actor.dump_dir=/data/tree/tree-data/gsm8k" ${LOG_FILE}
