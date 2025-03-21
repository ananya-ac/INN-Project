#!/bin/bash
if [ "$#" -lt 2 ]; then
  echo "Error: You must provide at least a prior_flag and one loss_function."
  echo "Usage: $0 <prior_flag> <loss_function1> <loss_function2> ..."
  exit 1
fi

PRIOR_FLAG=$1
LOSS_FUNCTIONS=("${@:2}")  # Get all loss functions passed as arguments

SCRIPT_PATH="train.py"

for LOSS_FUNCTION in "${LOSS_FUNCTIONS[@]}"
do
  for i in {1..10}
  do
    echo "Running $SCRIPT_PATH, iteration $i with Loss Function: $LOSS_FUNCTION, Prior: $PRIOR_FLAG, Dataset: ik_robotics"
    python3 $SCRIPT_PATH $LOSS_FUNCTION $PRIOR_FLAG 'ik_robotics'
  done
done
