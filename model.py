import torch
import valohai
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.nn import functional as F


class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self, lr_rate):
        super(LightningMNISTClassifier, self).__init__()
        self.save_hyperparameters()
        self.training_step_outputs = {"loss": [], "acc": []}
        self.validation_step_outputs = {"loss": [], "acc": []}
        self.test_step_outputs = {"loss": [], "acc": []}

        self.accuracy = Accuracy(num_classes=10, task="multiclass")

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)
        self.lr_rate = lr_rate

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 128)
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 256) -> (b, 10)
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.training_step_outputs["loss"].append(loss)
        self.training_step_outputs["acc"].append(acc)

        self.log_dict(
            {"train_loss": loss, "train_acc": acc},
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.validation_step_outputs["loss"].append(loss)
        self.validation_step_outputs["acc"].append(acc)
        self.log_dict(
            {"val_loss": loss, "val_acc": acc},
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return loss

    def on_train_epoch_end(self):
        with valohai.metadata.logger() as logger:
            train_loss = (
                torch.stack(self.training_step_outputs["loss"]).mean().cpu().item()
            )
            train_acc = (
                torch.stack(self.training_step_outputs["acc"]).mean().cpu().item()
            )
            logger.log("epoch", self.current_epoch + 1)
            logger.log("train_acc", train_acc)
            logger.log("train_loss", train_loss)

        for metric in self.training_step_outputs.values():
            metric.clear()

    def on_validation_epoch_end(self):
        with valohai.metadata.logger() as logger:
            train_loss = (
                torch.stack(self.validation_step_outputs["loss"]).mean().cpu().item()
            )
            train_acc = (
                torch.stack(self.validation_step_outputs["acc"]).mean().cpu().item()
            )
            logger.log("epoch", self.current_epoch + 1)
            logger.log("val_acc", train_acc)
            logger.log("val_loss", train_loss)

        for metric in self.validation_step_outputs.values():
            metric.clear()

    def on_test_epoch_end(self):
        with valohai.metadata.logger() as logger:
            test_loss = torch.stack(self.test_step_outputs["loss"]).mean().cpu().item()
            test_acc = torch.stack(self.test_step_outputs["acc"]).mean().cpu().item()
            logger.log("epoch", self.current_epoch + 1)
            logger.log("test_acc", test_acc)
            logger.log("test_loss", test_loss)

        for metric in self.test_step_outputs.values():
            metric.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
                        'name': 'expo_lr'}
        return [optimizer], [lr_scheduler]
