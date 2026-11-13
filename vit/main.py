import torch
from model import create_model
from train import train_model
from src.dataloaders import get_dataloaders

dataset_names = ["caltech", "voc"]
sizes = ["10mb", "100mb", "1gb"]

for ds_n in dataset_names:
    for sz in sizes:
        train_loader, val_loader = get_dataloaders(
            dataset_name = ds_n,
            size = sz,
            batch_size = 4
        )

        model = create_model(ds_n)
        model = train_model(ds_n, model, train_loader, val_loader)

