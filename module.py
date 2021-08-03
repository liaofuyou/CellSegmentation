import ssl

import albumentations
import cv2
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.utils.data
from torch import nn
from torch.optim import Adam
from torchmetrics import AverageMeter
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from dsb_datamodule import DSBDataModule
from metrics import iou_score

ssl._create_default_https_context = ssl._create_unverified_context


class CellSegmentationModule(pl.LightningModule):

    def __init__(self, num_classes):
        super().__init__()
        self.save_hyperparameters()

        self.net = deeplabv3_mobilenet_v3_large(pretrained=False,
                                                num_classes=num_classes)
        self.criterion = nn.BCEWithLogitsLoss()

        self.metric = AverageMeter()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        output = self(x)["out"]
        loss = self.criterion(output, y)

        self.log("Acc", self.metric(iou_score(output, y)), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def predict(self, image_path):
        transform = albumentations.Compose([
            albumentations.Resize(96, 96),
            albumentations.Normalize(),
        ])

        img = cv2.imread(image_path)
        img = transform(image=img)['image']
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img)

        self.eval()
        img = torch.unsqueeze(img, dim=0)
        output = self(img)["out"]
        output = output[0, 0, :, :]  # [1,1,96,96] TO [96,96]

        new = torch.zeros_like(output)
        new[output >= 0.5] = 1

        return new.detach().numpy()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


def predict():
    model = CellSegmentationModule(num_classes=1)
    out = model.predict("inputs/stage1_train/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9/images/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9.png")
    # print(out)
    plt.imshow(out)
    plt.show()


def cli_main():
    # init model
    model = CellSegmentationModule(num_classes=1)

    # init data
    dm = DSBDataModule()

    # train
    trainer = pl.Trainer(max_epochs=10,
                         val_check_interval=0.25,
                         fast_dev_run=False)
    trainer.fit(model, dm)


if __name__ == '__main__':
    cli_main()
