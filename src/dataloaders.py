
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import Caltech256, VOCDetection

from preprocessing import get_transform

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SPLIT_DIR = ROOT_DIR / "splits"

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"
]

VOC_CLASS_TO_INDEX = {
    name: index for index, name in enumerate(VOC_CLASSES)
}


class VOCClassification(Dataset):
    """Convert VOC detection annotations to multi-label targets."""

    def __init__(self, root, image_set, transform=None):
        self.dataset = VOCDetection(
            root=str(root),
            year="2012",
            image_set=image_set,
            download=False,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, annotation = self.dataset[index]

        objects = annotation["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]

        labels = torch.zeros(len(VOC_CLASSES), dtype=torch.float32)

        for obj in objects:
            class_name = obj["name"]
            if class_name in VOC_CLASS_TO_INDEX:
                labels[VOC_CLASS_TO_INDEX[class_name]] = 1.0

        return image, labels


def load_indices(filename):
    with open(SPLIT_DIR / filename) as f:
        return json.load(f)


def get_dataloaders(dataset_name, size, batch_size=32, num_workers=0):
    """
    Return training and evaluation DataLoaders.

    dataset_name: 'caltech' or 'voc'
    size: '10mb', '100mb', or '1gb'
    """

    if size not in {"10mb", "100mb", "1gb"}:
        raise ValueError("Size must be 10mb, 100mb, or 1gb")

    transform = get_transform()

    if dataset_name == "caltech":
        dataset = Caltech256(
            root=str(DATA_DIR),
            download=False,
            transform=transform
        )

        train = Subset(
            dataset,
            load_indices(f"caltech_{size}.json")
        )

        val = Subset(
            dataset,
            load_indices("caltech_test.json")
        )

    elif dataset_name == "voc":
        train_dataset = VOCClassification(
            DATA_DIR, "train", transform
        )

        train = Subset(
            train_dataset,
            load_indices(f"voc_{size}.json")
        )

        val = VOCClassification(
            DATA_DIR, "val", transform
        )

    else:
        raise ValueError("Dataset must be 'caltech' or 'voc'")

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader
