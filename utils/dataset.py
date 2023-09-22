from typing import Union

import numpy as np
import torch
from torchvision import transforms

from configs.settings import BASE_DIR

DATA_DIR = BASE_DIR / "data"


def load_dataset(args, train):
    match args.dataset.lower():
        # image dataset
        case "mnist":
            from torchvision.datasets import MNIST

            dataset = MNIST(
                root=DATA_DIR,
                train=train,
                download=True,
                transform=transforms.ToTensor(),
            )
        case "cifar10":
            from torchvision.datasets import CIFAR10

            dataset = CIFAR10(
                root=DATA_DIR,
                train=train,
                download=True,
                transform=transforms.ToTensor(),
            )
        case "imagenet":
            from torchvision.datasets import ImageNet

            dataset = ImageNet(
                root=DATA_DIR,
                train="train" if train else "val",
                transform=transforms.ToTensor(),
            )
        # text dataset
        case "imdb":
            from torchtext.datasets import IMDB

            dataset = IMDB(root=DATA_DIR, split="train" if train else "test")
        case "dbpedia":
            from torchtext.datasets import DBpedia

            dataset = DBpedia(root=DATA_DIR, split="train" if train else "test")
        # audio dataset
        case "speechcommands":
            # speech commands accept [None, training, validation, testing] by subset
            from torchaudio.datasets import SPEECHCOMMANDS

            dataset = SPEECHCOMMANDS(
                root=DATA_DIR,
                download=True,
                subset="training" if train else "testing",
            )
        # video dataset
        case "hmdb51":
            from torchvision.datasets import HMDB51

            dataset = HMDB51(
                root=DATA_DIR,
                annotation_path=DATA_DIR / "hmdv51_annotation",
            )
        case "kinetics":
            from torchvision.datasets import Kinetics

            dataset = Kinetics(
                root=DATA_DIR / "video_data" / "kinetics",
                frames_per_clip=100,
                num_classes="400",
                split="val",
                download=False,
                num_download_workers=20,
                num_workers=24,
            )
        case _:
            raise NotImplementedError("Dataset %s not support.", args.dataset)
    return dataset


def get_image_by_index(args, index, dataset=None, train=True):
    if dataset is None:
        dataset = load_dataset(args=args, train=train)
    return dataset[index]


def get_image_by_index(args, index, dataset=None, train=True):
    if dataset is None:
        dataset = load_dataset(args=args, train=train)
    return dataset[index]


class CleanDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        if isinstance(
            dataset, torch.utils.data.datapipes.iter.sharding.ShardingFilterIterDataPipe
        ):
            self.dataset = list(dataset)
        else:
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        x, y = data
        if isinstance(y, str):
            return x, y, 0, x
        else:
            return x, y, 0, y


class CleanAudioDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        waveform, sample_rate, label, speaker_id, utterance_number = data
        return (
            waveform,
            sample_rate,
            label,
            speaker_id,
            utterance_number,
            0,
            label,
        )


class CleanVideoDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        video, audio, label = data
        return (
            video,
            audio,
            label,
            0,
            label,
        )


class AddMaskPatchTrigger(object):
    def __init__(
        self,
        trigger_array: Union[np.ndarray, torch.Tensor],
    ):
        self.trigger_array = trigger_array

    def __call__(self, img, target=None, image_serial_id=None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return img * (self.trigger_array == 0) + self.trigger_array * (
            self.trigger_array > 0
        )


class SimpleAdditiveTrigger(object):
    def __init__(
        self,
        trigger_array: np.ndarray,
    ):
        self.trigger_array = trigger_array

    def __call__(self, img, target=None, image_serial_id=None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        img0 = np.transpose(np.array(img), [1, 2, 0])
        img1 = np.clip(img0.astype(float) + self.trigger_array / 255, 0, 1)
        return torch.tensor(np.transpose(img1, [2, 0, 1]),dtype= torch.float32)


class CleanIMDBWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        if isinstance(
            dataset, torch.utils.data.datapipes.iter.sharding.ShardingFilterIterDataPipe
        ):
            self.dataset = list(dataset)
        else:
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        label, text = data
        # IMDB start from 1
        label = label - 1
        if isinstance(text, str):
            return label, text, 0, label
        else:
            return label, text, 0, label
