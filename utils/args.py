import argparse
from typing import Tuple

import yaml


def init_args(parser: argparse.ArgumentParser, name=None) -> argparse.ArgumentParser:
    parser.add_argument("-n", "--num_workers", type=int, help="dataloader num_workers")
    parser.add_argument("--device", type=str)
    parser.add_argument(
        "--lr_scheduler", type=str, help="which lr_scheduler use for optimizer"
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dataset", type=str, help="which dataset to use")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--wd", type=float, help="weight decay of sgd")
    parser.add_argument("--client_optimizer", type=int)
    parser.add_argument("--random_seed", type=int, help="random_seed")
    parser.add_argument("--frequency_save", type=int, help="frequency_save, 0 is never")
    parser.add_argument("--model", type=str, help="choose which kind of model")
    parser.add_argument(
        "--save_folder_name",
        type=str,
        help="(Optional) should be time str + given unique identification str",
    )
    return parser


def add_yaml_to_args(args, path):
    with open(path, "r") as f:
        mix_defaults = yaml.safe_load(f)
    mix_defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = mix_defaults

    args.num_classes = get_num_classes(args.dataset)
    if args.dataset in ["cifar10", "gtsrb"]:
        # add dataset related info to args
        args.input_height, args.input_width, args.input_channel = get_input_shape(
            args.dataset
        )
        args.input_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"


def get_num_classes(dataset_name: str) -> int:
    # idea : given name, return the number of class in the dataset
    if dataset_name in ["mnist", "cifar10"]:
        num_classes = 10
    elif dataset_name == "gtsrb":
        num_classes = 43
    elif dataset_name == "celeba":
        num_classes = 8
    elif dataset_name == "cifar100":
        num_classes = 100
    elif dataset_name == "tiny":
        num_classes = 200
    elif dataset_name == "imagenet":
        num_classes = 1000
    elif dataset_name == "imdb":
        num_classes = 2
    elif dataset_name == "speechcommands":
        num_classes = 36
    elif dataset_name == "kinetics":
        num_classes = 400
    else:
        raise Exception("Invalid Dataset")
    return num_classes


def get_input_shape(dataset_name: str) -> Tuple[int, int, int]:
    # idea : given name, return the image size of images in the dataset
    if dataset_name == "cifar10":
        input_height = 32
        input_width = 32
        input_channel = 3
    elif dataset_name == "gtsrb":
        input_height = 32
        input_width = 32
        input_channel = 3
    elif dataset_name == "mnist":
        input_height = 28
        input_width = 28
        input_channel = 1
    elif dataset_name == "celeba":
        input_height = 64
        input_width = 64
        input_channel = 3
    elif dataset_name == "cifar100":
        input_height = 32
        input_width = 32
        input_channel = 3
    elif dataset_name == "tiny":
        input_height = 64
        input_width = 64
        input_channel = 3
    elif dataset_name == "imagenet":
        input_height = 224
        input_width = 224
        input_channel = 3
    else:
        raise Exception("Invalid Dataset")
    return input_height, input_width, input_channel
