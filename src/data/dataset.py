import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml


class DriverDataset(Dataset):
    def __init__(self, csv_path, config_path="src/configs/base.yaml", split="train"):
        self.df = pd.read_csv(csv_path)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.split = split
        self.task = self.config["task"]  # "levels3" o "classes10"
        self.transforms = self._build_transforms(split)

    def _build_transforms(self, split):
        size = self.config["image_size"]
        h, w = (size, size) if isinstance(size, int) else tuple(size)

        if split == "train":
            return A.Compose([
                A.Resize(256, 256),
                # ✅ Albumentations 2.x usa 'size' como tuple
                A.RandomResizedCrop(size=(h, w), scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                A.MotionBlur(p=0.2),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(256, 256),
                # ✅ CenterCrop ahora recibe directamente (h, w)
                A.CenterCrop(h, w),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row["filepath"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = row["label3"] if self.task == "levels3" else row["label10"]
        if self.transforms:
            image = self.transforms(image=image)["image"]
        return image, torch.tensor(label, dtype=torch.long)
