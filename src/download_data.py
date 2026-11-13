
from pathlib import Path
from torchvision.datasets import Caltech256, VOCDetection

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_caltech256():
    print("Downloading Caltech-256...")

    dataset = Caltech256(
        root=str(DATA_DIR),
        download=True
    )

    print(f"Caltech-256: {len(dataset)} images")
    return dataset


def download_pascal_voc():
    print("Downloading Pascal VOC 2012...")

    train = VOCDetection(
        root=str(DATA_DIR),
        year="2012",
        image_set="train",
        download=True
    )

    val = VOCDetection(
        root=str(DATA_DIR),
        year="2012",
        image_set="val",
        download=False
    )

    print(f"Pascal VOC training: {len(train)} images")
    print(f"Pascal VOC validation: {len(val)} images")

    return train, val


if __name__ == "__main__":
    download_caltech256()
    download_pascal_voc()
