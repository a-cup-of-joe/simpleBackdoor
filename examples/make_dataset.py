import argparse
import sys

sys.path.append("../")

import lightning as L
import torch
from torch.utils.data import DataLoader

from attacks.image import BadNet
from configs.settings import BASE_DIR
from models.base import ImageModelWrapper
from utils.args import add_yaml_to_args, init_args
from utils.dataset import Clean, load_dataset
from utils.model import load_model
from utils.save import get_log_folder_name

if __name__ == "__main__":
    # load args
    name = "badnet"
    parser = argparse.ArgumentParser()
    init_args(parser)
    args = parser.parse_args()
    add_yaml_to_args(args, BASE_DIR / "configs" / "default.yaml")

    # train bengin model and get acc
    train_set = load_dataset(args, train=True)
    benign_set = Clean(dataset=train_set)
    benign_loader = DataLoader(dataset=benign_set, batch_size=args.batch_size)
    test_set = load_dataset(args, train=False)
    clean_test_set = Clean(dataset=test_set)
    clean_loader = DataLoader(dataset=clean_test_set, batch_size=args.batch_size)

    original_model = load_model(args=args)
    lightning_model = ImageModelWrapper(model=original_model, args=args)

    # get poison training set
    poison_train_set = BadNet(dataset=train_set, mode="train", args=args)
    torch.save(
        poison_train_set,
    )
