from abc import ABC, abstractmethod

from torch.utils.data import Dataset
from tqdm import tqdm

import random

import numpy as np
import torch


class Base(ABC):
    """
    Base class for attack:
    - common function here
    - config default parameters
    - etc.
    """

    def __init__(self, dataset, args, mode="train") -> None:
        assert isinstance(dataset, Dataset), "dataset is an unsupported dataset type."
        assert mode in ["train", "test"]
        self.mode = mode
        self.dataset = dataset
        # load default args
        self.args = args
        self._set_seed()
        # get poison_list
        self.pratio = self.args.pratio
        self.poison_index = self._get_poison_index()

    @abstractmethod
    def _pop_original_class(self):
        ...

    @abstractmethod
    def make_poison_data(self, data):
        ...

    def _set_seed(self):
        seed = self.args.random_seed
        torch.manual_seed(seed)

        random.seed(seed)

        np.random.seed(seed)

    def _get_poison_index(self):
        pratio = self.args.pratio
        poison_index = dict()
        if pratio is not None or round(pratio * len(self.dataset)):
            poison_array = np.random.choice(
                np.arange(len(self.dataset)),
                round(pratio * len(self.dataset)),
                replace=False,
            )
            for idx in poison_array:
                poison_index[int(idx)] = 1
        return poison_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> torch.t_copy:
        # fetch a data point
        data = self.dataset[index]
        if index in self.poison_index or self.mode == "test":
            # poison
            return self.make_poison_data(data, index)
        else:
            # clean
            x, y = data
            is_poison = 0
            y_original = y
            return (x, y, is_poison, y_original)

    def make_and_save_dataset(self):
        all_poison_data = []
        print("making all poison datast:")
        for idx in tqdm(range(len(self.dataset))):
            data = self.dataset[idx]
            poison_data = self.make_poison_data(data, idx)
            all_poison_data.append(poison_data)
        from torch import save

        filename = "%s_%s_poison_dataset.pt" % (self.attack_type, self.attack_name)
        save_path = self.args.save_folder_name / filename
        save(all_poison_data, save_path.as_posix())
        print("dataset saved: %s" % self.args.save_folder_name)

class ImageBase(Base, Dataset):
    """
    Base class for attack:
    - common function here
    - config default parameters
    - etc.
    """

    def __init__(self, dataset, args) -> None:
        super().__init__(dataset, args)

    @abstractmethod
    def make_poison_data(self, img):
        ...
