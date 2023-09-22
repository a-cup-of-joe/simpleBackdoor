from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from attacks.base import ImageBase
from configs.settings import BASE_DIR
from utils.dataset import AddMaskPatchTrigger


class BadNet(ImageBase):
    def __init__(self, dataset, mode="train", args=None, save_img=True) -> None:
        super().__init__(dataset, args)
        self.attack_type = "image"
        self.attack_name = "BadNet"
        # self.save = save_img
        assert mode in ["train", "test"]
        self.mode = mode
        if self.mode == "test":
            # pop all the attack target when test
            self._pop_original_class()

    def _pop_original_class(self):
        classes = [i for i in range(10)]
        classes.pop(int(self.args.attack_target))
        classes = torch.tensor(classes)

        indices = (
            (torch.tensor(self.dataset.targets)[..., None] == classes)
            .any(-1)
            .nonzero(as_tuple=True)[0]
        )

        subset = torch.utils.data.Subset(self.dataset, indices)
        self.dataset = subset
        self.args.classes = classes

    def make_poison_data(self, data, index):
        # define poison image transformer
        trans = transforms.Compose(
            [transforms.Resize(self.args.input_size[:2]), transforms.ToTensor()]
        )
        trigger_path = Path(self.args.patch_mask_path)
        bd_transform = AddMaskPatchTrigger(trans(Image.open(BASE_DIR / trigger_path)))
        # poison the image data
        x, y = data
        x_poison = bd_transform(x)
        # set mislabel
        y_poison = self.args.attack_target
        is_poison = 1
        y_original = y
        # save img
        # if self.save:
        #     self.save_img(x_poison, y_poison, y_original, index)
        return (x_poison, y_poison, is_poison, y_original)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> torch.t_copy:
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

    def make_and_save_dataset(self, ratio):
        np.random.seed(0)
        poison_idxs = np.random.choice(len(self.dataset),int(len(self.dataset)*ratio), replace = False)
        mix_poison_data = []
        print("making all poison datast:")
        for idx in tqdm(range(len(self.dataset))):
            data = self.dataset[idx]
            if idx in poison_idxs:
                poison_data = self.make_poison_data(data, idx)
                mix_poison_data.append(poison_data)
            else:
                mix_poison_data.append((data[0], data[1], 0, data[1]))
        from torch import save

        filename = "%s_%s_%s_poison_dataset.pt" % (self.attack_type, self.attack_name, ratio)
        save_path = self.args.save_folder_name / filename
        save(mix_poison_data, save_path.as_posix())
        print("dataset saved: %s" % self.args.save_folder_name)