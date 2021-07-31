import pytorch_lightning as pl
from torch import nn
from torch.optim import Adam

from archs import NestedUNet
from dsb_datamodule import DSBDataModule


class UnetTrainer(pl.LightningModule):

    def __init__(self, num_classes, input_channels):
        super().__init__()
        self.save_hyperparameters()
        self.net = NestedUNet(num_classes, input_channels)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(input)
        loss = self.criterion(output, y)
        # iou = iou_score(output, y)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)


def cli_main():
    # init model
    model = UnetTrainer(num_classes=20, input_channels=3)

    # init data
    dm = DSBDataModule()

    # train
    trainer = pl.Trainer(max_epochs=10,
                         val_check_interval=0.25,
                         fast_dev_run=True)
    trainer.fit(model, dm)


if __name__ == '__main__':
    cli_main()
