import sys

from generate_specs import main as generate_spec



class args_object():
    def __init__(self, seed=42):
        self.dataset = "cifar10"
        self.max_epsilon = 0
        self.n = 36
        self.block = False
        self.seed = seed
        self.start_idx = None
        self.network = None
        self.instances = "../specs/instances.csv"
        self.new_instances = False
        self.time_out = 300.0
        self.search_eps = None

    def set(self, parameter_dic):
        for k, v in parameter_dic.items():
            if hasattr(self,k):
                setattr(self,k,v)


def main():
    assert len(sys.argv) == 2, "incorrect number of parameters"
    seed = sys.argv[1]
    print(f"Seed {seed} passed")

    args = args_object(int(seed))

    args.set({"max_epsilon": 0.035, "n": 72, "network": "../nets/resnet_3b2_bn_mixup_ssadv_4.0_bs128_lr-1_v2.pth",
              "search_eps": 0.7, "new_instance": True})
    generate_spec(args)

    # args.set({"max_epsilon": 0.03, "n": 2, "network": "../nets/resnet_3b2_bn_mixup_adv_4.0_bs128_lr-1.pth",
    #           "search_eps": 0.7, "new_instance": False})
    # generate_spec(args)


if __name__ == "__main__":
    main()