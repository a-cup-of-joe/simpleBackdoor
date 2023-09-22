import time
from typing import Any, Optional

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import Accuracy


class ImageModelWrapper(L.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.acc = Accuracy(task="multiclass", num_classes=self.args.num_classes)

    def on_train_epoch_start(self) -> None:
        self.time = time.time()
        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        x, y, is_poison, y_original = batch
        z = self.model(x)
        loss = F.cross_entropy(z, y)
        self.log("trian_loss", loss)
        return loss

    def on_train_epoch_end(self) -> None:
        train_time = time.time() - self.time
        self.log("training_time", train_time)
        return super().on_train_epoch_end()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, is_poison, y_original = batch
        z = self.model(x)
        loss = F.cross_entropy(z, y)

        clean_acc = self.acc(z, y_original)
        if dataloader_idx == 0:
            self.log("test_clean_acc", clean_acc)
            return loss, clean_acc
        if dataloader_idx == 1:
            ra = clean_acc
            asr = self.acc(z, y)
            self.log("test_ra", ra)
            self.log("test_asr", asr)
            return loss, ra, asr

    def configure_optimizers(self):
        if self.args.client_optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError(
                "Optimizer %s not supported." % self.args.client_optimizer
            )

        return optimizer


class TextModelWrapper(L.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.acc = Accuracy(task="multiclass", num_classes=self.args.num_classes)

    def on_train_epoch_start(self) -> None:
        self.time = time.time()
        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        label, text, is_poison, pre_label = batch
        inputs = self.args.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        # print(inputs)
        # print("device: ", self.model.device)
        inputs["labels"] = label
        inputs.to("cuda")
        ret = self.model(**inputs)
        # loss = F.cross_entropy(ret[0], label)
        loss = ret.loss
        self.log("trian_loss", loss)
        return loss

    def on_train_epoch_end(self) -> None:
        torch.cuda.empty_cache()
        train_time = time.time() - self.time
        self.log("training_time", train_time)
        return super().on_train_epoch_end()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        label, text, is_poison, pre_label = batch
        inputs = self.args.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        inputs.to("cuda")
        ret = self.model(**inputs)
        loss = ret.loss
        logits = ret[0]
        # loss = F.cross_entropy(z[0], label)

        # print(z[0].shape, label.shape, label, pre_label.shape, pre_label)
        clean_acc = self.acc(logits, pre_label)
        if dataloader_idx == 0:
            self.log("test_clean_acc", clean_acc)
            return loss, clean_acc
        if dataloader_idx == 1:
            ra = clean_acc
            asr = self.acc(logits, label)
            self.log("test_ra", ra)
            self.log("test_asr", asr)
            return loss, ra, asr

    def configure_optimizers(self):
        if self.args.client_optimizer == "adamw":
            model = self.model
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
            # from transformers import get_linear_schedule_with_warmup
            # scheduler = get_linear_schedule_with_warmup(
            #     optimizer,
            #     num_warmup_steps=self.hparams.warmup_steps,
            #     num_training_steps=self.trainer.estimated_stepping_batches,
            # )
            # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        else:
            raise NotImplementedError(
                "Optimizer %s not supported." % self.args.client_optimizer
            )

        # return [optimizer], [scheduler]
        return optimizer


class AudioModelWrapper(L.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.acc = Accuracy(task="multiclass", num_classes=self.args.num_classes)

    def label_to_index(self, word):
        # Return the position of the word in labels
        return torch.tensor(self.args.labels.index(word))

    def on_train_epoch_start(self) -> None:
        self.time = time.time()
        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        waveform, target, is_poison, label_original = batch
        preds = self.model(waveform)
        target = target.cuda()
        preds = preds.squeeze()
        loss = F.cross_entropy(preds, target)
        # loss = F.nll_loss(preds, target)
        self.log("trian_loss", loss)
        return loss

    def on_train_epoch_end(self) -> None:
        train_time = time.time() - self.time
        self.log("training_time", train_time)
        return super().on_train_epoch_end()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        waveform, target, is_poison, pre_target = batch
        preds = self.model(waveform)
        target = target.cuda()

        loss = F.cross_entropy(preds.squeeze(), target)

        preds = preds.squeeze().argmax(1)
        clean_acc = self.acc(preds, pre_target)
        if dataloader_idx == 0:
            self.log("test_clean_acc", clean_acc)
            return loss, clean_acc
        if dataloader_idx == 1:
            ra = clean_acc
            asr = self.acc(preds, target)
            self.log("test_ra", ra)
            self.log("test_asr", asr)
            return loss, ra, asr

    def configure_optimizers(self):
        if self.args.client_optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.args.lr, weight_decay=self.args.wd
            )
            # scheduler = torch.optim.lr_scheduler.StepLR(
            #     optimizer, step_size=20, gamma=0.1
            # )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.args.epochs
            )
        else:
            raise NotImplementedError(
                "Optimizer %s not supported." % self.args.client_optimizer
            )

        return [optimizer], [scheduler]


class VideoModelWrapper(L.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.acc = Accuracy(task="multiclass", num_classes=self.args.num_classes)

    def on_train_epoch_start(self) -> None:
        self.time = time.time()
        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        video, label, is_poison, pre_label = batch
        print("\n", label)
        preds = self.model(video)
        print(preds.argmax(1))
        label = label.cuda()
        loss = F.cross_entropy(preds, label)
        # loss = F.nll_loss(preds, label)
        self.log("trian_loss", loss)
        return loss

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> int | None:
        torch.cuda.empty_cache()
        return super().on_train_batch_start(batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        train_time = time.time() - self.time
        self.log("training_time", train_time)
        return super().on_train_epoch_end()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        video, label, is_poison, pre_label = batch
        preds = self.model(video)
        print("\n", label)
        print(preds.argmax(1))
        loss = F.cross_entropy(preds, label)

        clean_acc = self.acc(preds, pre_label)
        if dataloader_idx == 0:
            self.log("test_clean_acc", clean_acc)
            return loss, clean_acc
        if dataloader_idx == 1:
            ra = clean_acc
            asr = self.acc(preds, label)
            self.log("test_ra", ra)
            self.log("test_asr", asr)
            return loss, ra, asr

    def configure_optimizers(self):
        if self.args.client_optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.args.lr, weight_decay=self.args.wd
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.1
            )
        else:
            raise NotImplementedError(
                "Optimizer %s not supported." % self.args.client_optimizer
            )

        return [optimizer], [scheduler]
        # return optimizer
