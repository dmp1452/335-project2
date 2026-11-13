
import json
from pathlib import Path
from collections import Counter

from torchvision.datasets import Caltech256, VOCDetection

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLIT_DIR = ROOT / "splits"

SIZES = ["10mb", "100mb", "1gb"]

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"
]


def load_indices(filename):
    with open(SPLIT_DIR / filename) as f:
        return set(json.load(f))


def validate_caltech():
    dataset = Caltech256(root=str(DATA_DIR), download=False)

    train = load_indices("caltech_train.json")
    test = load_indices("caltech_test.json")

    print("\n=== Caltech Validation ===")

    assert train.isdisjoint(test)
    assert len(train | test) == len(dataset)

    previous = set()

    original_counts = Counter(dataset.y[i] for i in train)

    for size in SIZES:
        indices = load_indices(f"caltech_{size}.json")

        assert indices <= train
        assert previous <= indices

        counts = Counter(dataset.y[i] for i in indices)
        assert len(counts) == 257

        max_difference = max(
            abs(
                counts[c] / len(indices)
                - original_counts[c] / len(train)
            )
            for c in original_counts
        )

        print(
            f"{size}: {len(indices)} images, "
            f"257 classes, "
            f"max class proportion difference: "
            f"{max_difference:.4f}"
        )

        previous = indices

    print("Caltech validation passed.")


def extract_voc_labels(annotation):
    objects = annotation["annotation"].get("object", [])

    if isinstance(objects, dict):
        objects = [objects]

    return {
        obj["name"]
        for obj in objects
        if obj["name"] in VOC_CLASSES
    }


def validate_voc():
    dataset = VOCDetection(
        root=str(DATA_DIR),
        year="2012",
        image_set="train",
        download=False
    )

    print("\n=== Pascal VOC Validation ===")

    labels = {}
    for i in range(len(dataset)):
        _, annotation = dataset[i]
        labels[i] = extract_voc_labels(annotation)

    original_counts = Counter()
    for image_labels in labels.values():
        original_counts.update(image_labels)

    previous = set()

    for size in SIZES:
        indices = load_indices(f"voc_{size}.json")

        assert all(0 <= i < len(dataset) for i in indices)
        assert previous <= indices

        counts = Counter()
        for i in indices:
            counts.update(labels[i])

        assert len(counts) == 20

        max_difference = max(
            abs(
                counts[c] / len(indices)
                - original_counts[c] / len(dataset)
            )
            for c in VOC_CLASSES
        )

        print(
            f"{size}: {len(indices)} images, "
            f"20 classes, "
            f"max class frequency difference: "
            f"{max_difference:.4f}"
        )

        previous = indices

    print("Pascal VOC validation passed.")


if __name__ == "__main__":
    validate_caltech()
    validate_voc()
