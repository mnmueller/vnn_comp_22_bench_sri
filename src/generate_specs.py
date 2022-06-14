import argparse
import numpy as np
from torchvision import transforms, datasets
import os
import onnxruntime as rt
import torch
from pgd_attack import pgd_attack
import resnetseq as models


def write_vnn_spec(dataset, index, eps, dir_path="./", prefix="spec", data_lb=0, data_ub=1, n_class=10, mean=None, std=None):
    x, y = dataset[index]
    x_lb = np.clip(x - eps, data_lb, data_ub)
    x_ub = np.clip(x + eps, data_lb, data_ub)

    if mean is not None and std is not None:
        mean = mean.detach().cpu().numpy()
        std = std.detach().cpu().numpy()
        x_lb = ((x_lb - mean) / std)
        x_ub = ((x_ub - mean) / std)

    x_lb = np.array(x_lb).reshape(-1)
    x_ub = np.array(x_ub).reshape(-1)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    spec_name = f"{prefix}_idx_{index}_eps_{eps:.5f}.vnnlib"
    spec_path = os.path.join(dir_path, spec_name)

    with open(spec_path, "w") as f:
        f.write(f"; Spec for sample id {index} and epsilon {eps:.5f}\n")

        f.write(f"\n; Definition of input variables\n")
        for i in range(len(x_ub)):
            f.write(f"(declare-const X_{i} Real)\n")

        f.write(f"\n; Definition of output variables\n")
        for i in range(n_class):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write(f"\n; Definition of input constraints\n")
        for i in range(len(x_ub)):
            f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
            f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n")

        f.write(f"\n; Definition of output constraints\n")
        f.write(f"(assert (or\n")
        for i in range(n_class):
            if i == y: continue
            f.write(f"\t(and (>= Y_{i} Y_{y}))\n")
        f.write(f"))\n")
    return spec_name


def get_sample_idx(n, block=True, seed=42, n_max=10000, start_idx=None):
    np.random.seed(seed)
    assert n <= n_max, f"only {n_max} samples are available"
    if block:
        if start_idx is None:
            start_idx = np.random.choice(n_max,1,replace=False)
        else:
            start_idx = start_idx % n_max
        idx = list(np.arange(start_idx,min(start_idx+n,n_max)))
        idx += list(np.arange(0,n-len(idx)))
    else:
        idx = list(np.random.choice(n_max,n,replace=False))
    return idx


def get_cifar10():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
    return datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.ToTensor())


def get_mnist():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
    return datasets.MNIST(data_path, train=False, download=True, transform=transforms.ToTensor())


def parse_args():
    parser = argparse.ArgumentParser(description='VNN specs', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, required=True, choices=["mnist", "cifar10"], help='The dataset to generate specs for')
    parser.add_argument('--max_epsilon', type=float, required=True, help='The epsilon for L_infinity perturbation')
    parser.add_argument('--n', type=int, default=25, help='The number of specs to generate')
    parser.add_argument('--block', action="store_true", default=False, help='Generate specs in a block')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for idx generation')
    parser.add_argument('--start_idx', type=int, default=None, help='Enforce block mode and return deterministic indices')
    parser.add_argument("--network", type=str, default=None, help="Network to evaluate as .onnx file.")
    parser.add_argument("--instances", type=str, default="../specs/instances.csv", help="Path to instances file")
    parser.add_argument("--new_instances", action="store_true", default=False, help="Overwrite old instances.csv")
    parser.add_argument('--time_out', type=float, default=300.0, help='the mean used to normalize the data with')
    parser.add_argument('--search_eps', type=float, default=None, help='Ratio for eps search')

    args = parser.parse_args()
    return args

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.start_idx is not None:
        args.block = True
        print(f"Generating {args.n} deterministic specs starting from index {args.start_idx}.")
    else:
        print(f"Generating {args.n} random specs using seed {args.seed}.")

    if args.dataset == "mnist":
        dataset = get_mnist()
        mean = [0.0]
        std = [1.0]
    elif args.dataset == "cifar10":
        dataset = get_cifar10()
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    else:
        assert False, "Unkown dataset" # Should be unreachable
    mean = torch.tensor(mean).view((1, -1, 1, 1)).to(dtype=torch.float32, device=device)
    std = torch.tensor(std).view((1, -1, 1, 1)).to(dtype=torch.float32, device=device)

    if args.network is not None:
        name_split = args.network.split("_")
        model_name = f"{name_split[0].split('/')[-1]}{name_split[1]}"
        bn = "bn" in name_split

        model = models.Models[model_name](in_ch=3, in_dim=32, bn=bn)
        state_dict = torch.load(args.network, map_location=device)['state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(device)
        onnx_path = args.network[:-4] + ".onnx"
        if not os.path.exists(onnx_path):
            torch.onnx.export(model,  torch.tensor(dataset[0][0].unsqueeze(0), device=device, dtype=torch.float32),
                  onnx_path, verbose=True, input_names=["input"], output_names=["output"])

    idxs = get_sample_idx(args.n, block=args.block, seed=args.seed, n_max=len(dataset), start_idx=args.start_idx)
    spec_path_rel = os.path.join("specs", args.dataset)
    spec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", spec_path_rel)

    instances_dir = os.path.dirname(args.instances)
    if not os.path.isdir(instances_dir):
        os.mkdir(instances_dir)

    i = 0
    ii = 1
    with open(args.instances, "w" if args.new_instances else "a") as f:
        while i<len(idxs):
            idx = idxs[i]
            i += 1
            if args.network is not None:
                x, y = dataset[idx]
                x = x.unsqueeze(0).numpy().astype(np.float32)
                x, y = torch.tensor(x, device=device, dtype=torch.float32), torch.tensor([y], device=device, dtype=torch.long)
                x = (x-mean)/std
                pred = model(x)
                y_pred = torch.argmax(pred, axis=-1)
                if all(y == y_pred) and args.search_eps is not None:
                    eps = args.max_epsilon / args.search_eps
                    eps_last = 0
                    while abs(eps-eps_last)/args.max_epsilon > 1e-3:
                        adv_found = pgd_attack("CIFAR" if dataset is "cifar10" else "MNIST", model, x, eps, data_min=-mean/std, data_max=(1-mean)/std, y=y, initialization="uniform")
                        eps_tmp = eps
                        if adv_found:
                            eps = eps - abs(eps-eps_last)/2
                        else:
                            if eps == args.max_epsilon / args.search_eps:
                                break
                            eps = eps + abs(eps-eps_last)/2
                        eps_last = eps_tmp
                    eps = args.search_eps * eps
                else:
                    eps = args.max_epsilon
            else:
                eps = args.max_epsilon

            if args.network is None or all(y == y_pred):
                print(f"Eps for sample {idx} set to {eps}")
                print(f"Model prediction {pred}")
                spec_i = write_vnn_spec(dataset, idx, eps, dir_path=spec_path, prefix=args.dataset + "_spec", data_lb=0, data_ub=1, n_class=10, mean=mean, std=std)
                f.write(f"{''if args.network is None else os.path.join('nets',os.path.basename(onnx_path))}, {os.path.join(spec_path_rel, spec_i)}, {args.time_out:.1f}\n")
                print(f"Instance added:")
                print(f"{''if args.network is None else os.path.join('nets',os.path.basename(onnx_path))}, {os.path.join(spec_path_rel, spec_i)}, {args.time_out:.1f}")
            else:
                print(f"Sample {idx} skipped as it was misclassified")
                if len(idxs) < len(dataset): # only sample idxs while there are still new samples to be found
                    if args.block: # if we want samples in a block, just get the next one
                        idxs.append(*get_sample_idx(1, True, n_max=len(dataset), start_idx=idxs[-1]+1))
                    else: # otherwise sample deterministicly (for given seed) until we find a new sample
                        tmp_idx = get_sample_idx(1, False, seed=args.seed+ii, n_max=len(dataset))
                        ii += 1
                        while tmp_idx in idxs:
                            tmp_idx = get_sample_idx(1, False, seed=args.seed + ii, n_max=len(dataset))
                            ii += 1
                        idxs.append(*tmp_idx)
        print(f"{len(idxs)-args.n} samples were misclassified and replacement samples drawn.")

if __name__ == "__main__":
    args = parse_args()
    main(args)