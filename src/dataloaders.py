
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import Caltech256, VOCDetection

from .preprocessing import get_transform
from .dataset_metadata import CALTECH_CLASSES, VOC_CLASSES

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SPLIT_DIR = ROOT_DIR / "splits"

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


def caltech_indices(dataset, filename):
    # Starter manifests may still list the last (clutter) folder.
    return [i for i in load_indices(filename) if dataset.y[i] < CALTECH_CLASSES]


def get_dataloaders(
    dataset_name, size, batch_size=32, num_workers=0, *,
    seed=None, pin_memory=False, tuning=False,
):
    """
    Return training and evaluation DataLoaders.

    dataset_name: 'caltech' or 'voc'
    size: '10mb', '100mb', or '1gb'
    """

    if size not in {"10mb", "100mb", "1gb"}:
        raise ValueError("Size must be 10mb, 100mb, or 1gb")

    if tuning and size != "10mb":
        raise ValueError("The tuning holdout is defined for the 10 MB subset")

    transform = get_transform()

    if dataset_name == "caltech":
        dataset = Caltech256(
            root=str(DATA_DIR),
            download=False,
            transform=transform
        )

        train = Subset(
            dataset,
            caltech_indices(dataset, f"caltech_{size}.json")
        )

        val = Subset(
            dataset,
            caltech_indices(dataset, "caltech_tune.json" if tuning else "caltech_test.json")
        )

    elif dataset_name == "voc":
        train_dataset = VOCClassification(
            DATA_DIR, "train", transform
        )

        train = Subset(
            train_dataset,
            load_indices(f"voc_{size}.json")
        )

        if tuning:
            val = Subset(train_dataset, load_indices("voc_tune.json"))
        else:
            val = VOCClassification(DATA_DIR, "val", transform)

    else:
        raise ValueError("Dataset must be 'caltech' or 'voc'")

    if tuning and not set(train.indices).isdisjoint(val.indices):
        raise ValueError("Tuning holdout overlaps the training subset. Regenerate splits.")

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=None if seed is None else torch.Generator().manual_seed(seed)
    )

    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader
