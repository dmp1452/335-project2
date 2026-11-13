
from src.dataloaders import get_dataloaders

for dataset_name in ["caltech", "voc"]:
    for size in ["10mb", "100mb", "1gb"]:
        train_loader, val_loader = get_dataloaders(
            dataset_name=dataset_name,
            size=size,
            batch_size=16
        )

        images, labels = next(iter(train_loader))

        print(f"\n{dataset_name} - {size}")
        print(f"Train images: {len(train_loader.dataset)}")
        print(f"Validation images: {len(val_loader.dataset)}")
        print(f"Batch shape: {images.shape}")
        print(f"Label shape: {labels.shape}")
