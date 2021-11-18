from __future__ import absolute_import, division

import torch

import cv2
import numpy as np
import pandas as pd
from torchvision import transforms

from torch.utils.data import Dataset

CLASSES = ["chandler", "joey", "monica", "phoebe", "rachel", "ross"]


class ImageDataset(Dataset):
    def __init__(self, train, test):
        labels_df = pd.read_csv("../data/processed/labels.csv")
        rows_with_no_class = labels_df[~(
            labels_df[CLASSES] == 0).all(axis=1)].copy()
        self.csv = rows_with_no_class
        self.train = train
        self.test = test
        self.all_image_names = self.csv[:]["filename"]
        self.all_labels = np.array(self.csv.drop(["filename", ], axis=1))
        self.train_ratio = int(0.6 * len(self.csv))
        self.valid_ratio = int(0.2 * len(self.csv))
        self.test_ratio = len(self.csv) - self.train_ratio - self.valid_ratio

        if self.train == True:
            print(f"Number of training images: {self.train_ratio}")
            self.image_names = list(self.all_image_names[: self.train_ratio])
            self.labels = list(self.all_labels[: self.train_ratio])

        elif self.train == False and self.test == False:
            print(f"Number of validation images: {self.valid_ratio}")
            self.image_names = list(
                self.all_image_names[
                    self.train_ratio: self.train_ratio + self.valid_ratio
                ]
            )
            self.labels = list(
                self.all_labels[self.train_ratio: self.train_ratio +
                                self.valid_ratio]
            )

        elif self.test == True and self.train == False:
            print(f"Number of test images: {self.test_ratio}")
            self.image_names = list(self.all_image_names[-self.test_ratio:])
            self.labels = list(self.all_labels[-self.test_ratio:])

        self.transform = transforms.Compose([transforms.ToTensor(), ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = cv2.imread(f"../data/raw_images/{self.image_names[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.transform(image)
        targets = self.labels[index]

        return {
            "image": image.clone().detach(),
            "label": torch.tensor(targets, dtype=torch.float32),
        }
