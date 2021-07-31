import os
from glob import glob
from typing import Optional

import albumentations
# pip install PyYaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dsb_dataset import DSBDataset


class DSBDataModule(LightningDataModule):

    def __init__(
            self,
            data_dir: str = "inputs/dsb2018_96/",
            num_workers: int = 1,
            batch_size: int = 32,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_train = ...
        self.dataset_val = ...

        # self.prepare_data()
        # self.setup()

    @property
    def num_classes(self):
        return 1

    def setup(self, stage: Optional[str] = None):
        # Data loading code
        img_ids = glob(os.path.join('inputs/dsb2018_96/images', '*.png'))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

        train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
        # 数据增强：
        train_transform = Compose([
            # transforms.RandomRotate90(),
            transforms.Flip(),
            OneOf([
                transforms.HueSaturationValue(),
                transforms.RandomBrightness(),
                transforms.RandomContrast(),
            ], p=1),  # 按照归一化的概率选择执行哪一个
            albumentations.Resize(96, 96),
            transforms.Normalize(),
        ])

        val_transform = Compose([
            albumentations.Resize(96, 96),
            transforms.Normalize(),
        ])

        self.dataset_train = DSBDataset(
            img_ids=train_img_ids,
            img_dir=os.path.join(self.data_dir, 'images'),
            mask_dir=os.path.join(self.data_dir, 'masks'),
            img_ext=".png",
            mask_ext=".png",
            num_classes=1,
            transform=train_transform)

        self.dataset_val = DSBDataset(
            img_ids=val_img_ids,
            img_dir=os.path.join(self.data_dir, 'images'),
            mask_dir=os.path.join(self.data_dir, 'masks'),
            img_ext=".png",
            mask_ext=".png",
            num_classes=1,
            transform=val_transform)

    def train_dataloader(self):
        """MNIST train set removes a subset to use for validation"""
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        """MNIST val set uses a subset of the training set for validation"""
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader
