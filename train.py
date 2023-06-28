import json
import os
import pytorch_lightning as pl
import valohai
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from model import LightningMNISTClassifier
from prepare_data import prepare_data


class Trainer:
    def __init__(
            self,
            model,
            dataloaders,
    ):
        self.model = model
        self.train_loader, self.val_loader = dataloaders
        self.model_path = valohai.outputs("models").dir_path
        self.best_checkpoint_path = None

    def train(self):
        callbacks = self.configure_callbacks()
        trainer = pl.Trainer(max_epochs=30, accelerator='gpu', callbacks=callbacks,
                             default_root_dir=self.model_path)
        trainer.fit(self.model, train_dataloaders=self.train_loader, val_dataloaders=self.val_loader)

        self.best_checkpoint_path = trainer.checkpoint_callback.best_model_path

    def configure_callbacks(self):
        # Set Early Stopping
        early_stopping = EarlyStopping('val_loss', mode='min', patience=3)
        # saves checkpoints to 'model_path' whenever 'val_loss' has a new min
        checkpoint_callback = ModelCheckpoint(dirpath=self.model_path,
                                              monitor='val_loss', mode='min', save_top_k=3)
        return [early_stopping, checkpoint_callback]

    def save_best_model(self):
        # Get the best checkpoint path and save the best_model.pt with an alias
        model = self.model.load_from_checkpoint(self.best_checkpoint_path)
        model.eval()

        checkpoint_name = os.path.basename(self.best_checkpoint_path)[:-4] + 'pt'
        model_output_path = valohai.outputs().path(f"models/{checkpoint_name}")
        torch.save(model.state_dict(), model_output_path)

        metadata_path = valohai.outputs().path(f"models/{checkpoint_name}.metadata.json")
        metadata = {
            "valohai.alias": 'best-model',
        }
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile)


if __name__ == '__main__':
    classifier = LightningMNISTClassifier(lr_rate=1e-3)

    train, val, test = prepare_data()
    mnist_dataloaders = DataLoader(train, batch_size=64), DataLoader(val, batch_size=64)

    trainer = Trainer(classifier, mnist_dataloaders)
    trainer.train()
    trainer.save_best_model()
