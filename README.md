VNNComp Benchmark suggestions
--------------------

Two benchmarks are proposed here, both on FFN, one using ReLU and the other Sigmoid activations.
The networks are available in `./nets`.

Evaluating 36 samples each for robustness against all non-label classes (using either a disjunctive or 9 separate properties), with a per sample timeout of 300s is suggested.
For the FFN we have `--epsilon 0.015` and for the Sigmoid network `--epsilon 0.012`.

After creating a conda or virtual environment with python 3.6 and installing requirements (`pip install -i requirements.txt`), new specs can be generated as follows.
To generate a set of deterministic test specs starting with test set index 0 and drawing until 36 samples are found that are classified correctly by the corresponding networks.
```
bash specs_from_seed.sh
```
To generate a set of specs based on randomly selected images, which are classified correctly by the corresponding models, additionally pass a random seed.
```
bash specs_from_seed.sh <seed>
```

To ensure the provided network files are evaluated correctly in any tool, we provide `evaluate_network.py` to evaluate (clean) samples.
```
python3 ./src/evaluate_network.py --dataset mnist --n 36 --seed 30 --start_idx 0 --network ./nets/mnist_relu_9_200.onnx --debug
```
Where `--start_idx` can be used to generate a block of `--n` non-random specs starting at `--start_idx`.
If `--start_idx` is not provided as argument but block is, a random start index is chosen and `--n` consecutive samples are used to generate specs.
If neither `--start_idx` nor `--block` are passed, `--n` random samples are chosen to generate specs.

For example evaluates the FFN network on the samples used to generate specs with the command before, printing all logits.