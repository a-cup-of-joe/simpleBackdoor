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
from utils.dataset import CleanDatasetWrapper, load_dataset
from utils.model import load_model
from utils.save import get_log_folder_name

import numpy as np

if __name__ == "__main__":
    # load args
    name = "badnet"
    parser = argparse.ArgumentParser()
    init_args(parser)
    args = parser.parse_args()
    add_yaml_to_args(args, BASE_DIR / "configs" / "attacks" / "image" / "badnet.yaml")

    # train benign model and get acc
    train_set = load_dataset(args, train=True)
    benign_set = CleanDatasetWrapper(dataset=train_set)
    benign_loader = DataLoader(dataset=benign_set, batch_size=args.batch_size)
    test_set = load_dataset(args, train=False)
    clean_test_set = CleanDatasetWrapper(dataset=test_set)
    clean_loader = DataLoader(dataset=clean_test_set, batch_size=args.batch_size)

    original_model = load_model(args=args)
    lightning_model = ImageModelWrapper(model=original_model, args=args)

    benign_log_path = get_log_folder_name(args)/"benign"
    args.save_folder_name = benign_log_path

    # trainer = L.Trainer(
    #     devices=1,
    #     max_epochs=args.epochs,
    #     default_root_dir=benign_log_path,
    #     fast_dev_run=False,
    #     log_every_n_steps=args.frequency_save,
    # )

    # # train benign model
    # trainer.fit(model=lightning_model, train_dataloaders=benign_loader)
    # # scripted_model = torch.jit.script(original_model)
    # # scripted_model.save(benign_log_path / "benign_scripted_model.pt")
    # torch.save(original_model.state_dict(), benign_log_path / "benign_scripted_model.pth")
    # # test on benign set
    # trainer.test(model=lightning_model, dataloaders=clean_loader)
    # get poison training set
    poison_train_set = BadNet(dataset=train_set, mode="train", args=args)
    poison_loader = DataLoader(dataset=poison_train_set, batch_size=args.batch_size)

    attack_log_path = get_log_folder_name(args)
    args.save_folder_name = attack_log_path
    # save poison training set
    torch.save(poison_train_set, attack_log_path / "poison_train_set.pt")
    optimizer = torch.optim.Adam(original_model.parameters(),lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss().to(torch.device("cuda"))
    # get model
    poison_test_set = BadNet(dataset=test_set, mode="test", args=args)
    original_model = load_model(args=args)
    for epoch in range(5):
        original_model.train()
        train_epoch_loss = []
        for idx,(data_x,data_y, is_poison, org_y) in enumerate(poison_loader):
            data_x = data_x.to(torch.device("cuda"))
            data_y = data_y.to(torch.device("cuda"))
            outputs = original_model(data_x)
            optimizer.zero_grad()
            loss = criterion(outputs, data_y)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
        print(np.average(train_epoch_loss))
    torch.save(original_model.state_dict(), attack_log_path / "poison_scripted_model.pth")

    # trainer = L.Trainer(
    #     devices=1,
    #     max_epochs=args.epochs,
    #     default_root_dir=attack_log_path,
    #     fast_dev_run=False,
    #     log_every_n_steps=args.frequency_save,
    # )
    # # train backdoor model
    # trainer.fit(model=lightning_model, train_dataloaders=poison_loader)
    # # scripted_model = torch.jit.script(original_model)
    # # scripted_model.save(attack_log_path / "poison_scripted_model.pt")
    # torch.save(lightning_model.model.state_dict(), attack_log_path / "poison_scripted_model.pth")
    # trainer.save_checkpoint(attack_log_path /"test.ckpt")

    # test on poison dataset
    poison_test_set = BadNet(dataset=test_set, mode="test", args=args)
    poison_loader = DataLoader(dataset=poison_test_set, batch_size=args.batch_size)
    # save poison test set
    torch.save(poison_test_set, attack_log_path / "poison_test_set.pt")
    test_trainer = L.Trainer(
        devices=1,
        num_nodes=1,
        max_epochs=args.epochs,
        default_root_dir=attack_log_path,
        fast_dev_run=False,
        log_every_n_steps=args.frequency_save,
    )
    test_trainer.test(model=lightning_model, dataloaders=[clean_loader, poison_loader])
