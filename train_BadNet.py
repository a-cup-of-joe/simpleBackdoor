import argparse
import sys
from pathlib import Path
import lightning as L
import torch
from torch.utils.data import DataLoader

sys.path.append(".")
from attacks import BadNet
from configs.settings import BASE_DIR
from models.base import ImageModelWrapper
from utils.args import add_yaml_to_args, init_args
from utils.dataset import CleanDatasetWrapper, load_dataset
from utils.model import load_model
from utils.save import get_log_folder_name
from tqdm import tqdm

def save_dataset(dataset, mode = "train"):
    datas = []
    for idx in tqdm(range(len(dataset.dataset))):
        data = dataset[idx]
        datas.append(data)
    if mode == "full":
        filename = "full_poison_train_dataset.pt"
    elif hasattr(dataset, "pratio"):
        filename = "%s_%s_%s_dataset.pt" % (dataset.attack_name,dataset.pratio,dataset.mode)
    else:
        filename = "%s_%s_dataset.pt" % ("clean", mode)
    save_path = get_log_folder_name(args)/"data"
    if not Path.exists(save_path):
        Path.mkdir(save_path)
    torch.save(datas, (save_path/filename).as_posix())
    print("dataset saved: %s" % save_path)

if __name__ == "__main__":
    # load args
    name = "badnet"
    parser = argparse.ArgumentParser()
    init_args(parser)
    args = parser.parse_args()
    add_yaml_to_args(args, BASE_DIR / "configs" / "badnet.yaml")

    # train benign model and get acc
    train_set = load_dataset(args, train=True)
    benign_set = CleanDatasetWrapper(dataset=train_set)
    benign_loader = DataLoader(dataset=benign_set, batch_size=args.batch_size)
    poison_train_set = BadNet(dataset=train_set, mode="train", args=args)
    poison_loader = DataLoader(dataset=poison_train_set, batch_size=args.batch_size)

    test_set = load_dataset(args, train=False)
    clean_test_set = CleanDatasetWrapper(dataset=test_set)
    clean_loader = DataLoader(dataset=clean_test_set, batch_size=args.batch_size)
    poison_test_set = BadNet(dataset=test_set, mode="test", args=args)
    poison_test_loader = DataLoader(dataset=poison_test_set, batch_size=args.batch_size)

    full_poison_train_set = BadNet(dataset=train_set, mode="test", args=args)

    save_dataset(benign_set)
    save_dataset(poison_train_set)
    save_dataset(clean_test_set, mode="test")
    save_dataset(poison_test_set)
    save_dataset(full_poison_train_set, mode="full")

    original_model = load_model(args=args)
    lightning_model = ImageModelWrapper(model=original_model, args=args)

    benign_log_path = get_log_folder_name(args)/"benign"
    args.save_folder_name = benign_log_path

    trainer = L.Trainer(
        devices=1,
        max_epochs=args.epochs,
        default_root_dir=benign_log_path,
        fast_dev_run=False,
        log_every_n_steps=args.frequency_save,
    )

    trainer.fit(model=lightning_model, train_dataloaders=benign_loader)
    torch.save(original_model.state_dict(), benign_log_path / "benign_scripted_model.pth")
    trainer.test(model=lightning_model, dataloaders=clean_loader)


    # get model
    original_model = load_model(args=args)
    lightning_model = ImageModelWrapper(original_model, args=args)
    attack_log_path = get_log_folder_name(args)
    args.save_folder_name = attack_log_path
    trainer = L.Trainer(
        devices=1,
        max_epochs=args.epochs,
        default_root_dir=attack_log_path,
        fast_dev_run=False,
        log_every_n_steps=args.frequency_save,
    )
    # train backdoor model
    trainer.fit(model=lightning_model, train_dataloaders=poison_loader)
    torch.save(lightning_model.model.state_dict(), attack_log_path / "poison_scripted_model.pth")

    test_trainer = L.Trainer(
        devices=1,
        num_nodes=1,
        max_epochs=args.epochs,
        default_root_dir=attack_log_path,
        fast_dev_run=False,
        log_every_n_steps=args.frequency_save,
    )
    test_trainer.test(model=lightning_model, dataloaders=[clean_loader, poison_test_loader])