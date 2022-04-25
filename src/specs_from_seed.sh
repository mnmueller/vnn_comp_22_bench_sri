#!/bin/bash

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $SCRIPT_PATH
GENERATE_PATH=$SCRIPT_PATH"/generate_specs.py"
echo $GENERATE_PATH

if [ "$#" == 1 ]; then
    python3 $GENERATE_PATH --dataset mnist --n 36 --seed $1 --epsilon 0.015 --network ../nets/mnist_relu_9_200.onnx --new_instances
    python3 $GENERATE_PATH --dataset mnist --n 36 --seed $(expr $1 + $1 + 1) --epsilon 0.012 --network ../nets/ffnnSIGMOID__Point_6x200.onnx
    exit 0
elif [ "$#" == 0 ]; then
    python3 $GENERATE_PATH --dataset mnist --n 36 --start_idx 0 --epsilon 0.015 --network ../nets/mnist_relu_9_200.onnx --new_instances
    python3 $GENERATE_PATH --dataset mnist --n 36 --start_idx 0 --epsilon 0.012 --network ../nets/ffnnSIGMOID__Point_6x200.onnx
    exit 0
else
  echo "Expected either no argument for deterministic generation or just a random seed. (got $# arguments)"
  exit 1
fi

