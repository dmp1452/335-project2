
import json
from pathlib import Path
from collections import Counter

from torchvision.datasets import Caltech256, VOCDetection

if __package__:
    from .dataset_metadata import (
        CALTECH_CLASSES, SIZE_BYTES, VOC_CLASSES, caltech_image_path, duplicate_images,
        voc_labels_from_xml,
    )
else:
    from dataset_metadata import (
        CALTECH_CLASSES, SIZE_BYTES, VOC_CLASSES, caltech_image_path, duplicate_images,
        voc_labels_from_xml,
    )

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLIT_DIR = ROOT / "splits"

SIZES = ["10mb", "100mb", "1gb"]

def load_indices(filename):
    with open(SPLIT_DIR / filename) as f:
        indices = json.load(f)
    assert len(set(indices)) == len(indices), f"Duplicate indices in {filename}"
    return set(indices)


def validate_caltech():
    dataset = Caltech256(root=str(DATA_DIR), download=False)

    train = load_indices("caltech_train.json")
    test = load_indices("caltech_test.json")

    print("\n=== Caltech Validation ===")

    assert train.isdisjoint(test)
    eligible = {i for i, label in enumerate(dataset.y) if label < CALTECH_CLASSES}
    assert (train | test) <= eligible
    duplicates = duplicate_images(
        ((i, caltech_image_path(dataset, i)) for i in eligible - test),
        (caltech_image_path(dataset, i) for i in test),
    )
    assert eligible - test - train == duplicates
    assert train.isdisjoint(duplicates)
    assert {dataset.y[i] for i in test} == set(range(CALTECH_CLASSES))
    tune = load_indices("caltech_tune.json")
    assert tune <= train and tune.isdisjoint(load_indices("caltech_10mb.json"))
    assert {dataset.y[i] for i in tune} == set(range(CALTECH_CLASSES))
    assert not duplicate_images(
        ((i, caltech_image_path(dataset, i)) for i in tune),
        (caltech_image_path(dataset, i) for i in load_indices("caltech_10mb.json")),
    )

    previous = set()

    original_counts = Counter(dataset.y[i] for i in train)

    for size in SIZES:
        indices = load_indices(f"caltech_{size}.json")

        assert indices <= train
        assert previous <= indices

        counts = Counter(dataset.y[i] for i in indices)
        assert len(counts) == CALTECH_CLASSES
        total_bytes = sum(caltech_image_path(dataset, i).stat().st_size for i in indices)
        assert total_bytes <= SIZE_BYTES[size]

        max_difference = max(
            abs(
                counts[c] / len(indices)
                - original_counts[c] / len(train)
            )
            for c in original_counts
        )

        print(
            f"{size}: {len(indices)} images, "
            f"{CALTECH_CLASSES} classes, {total_bytes:,}/{SIZE_BYTES[size]:,} JPEG bytes, "
            f"max class proportion difference: "
            f"{max_difference:.4f}"
        )

        previous = indices

    print("Caltech validation passed.")


def validate_voc():
    dataset = VOCDetection(
        root=str(DATA_DIR),
        year="2012",
        image_set="train",
        download=False
    )

    val = VOCDetection(root=str(DATA_DIR), year="2012", image_set="val")
    assert set(dataset.images).isdisjoint(val.images)
    assert not duplicate_images(enumerate(dataset.images), val.images)
    print("\n=== Pascal VOC Validation ===")

    labels = {i: voc_labels_from_xml(path) for i, path in enumerate(dataset.annotations)}
    assert set().union(*(voc_labels_from_xml(path) for path in val.annotations)) == set(VOC_CLASSES)
    tune = load_indices("voc_tune.json")
    assert tune <= labels.keys() and tune.isdisjoint(load_indices("voc_10mb.json"))
    assert set().union(*(labels[i] for i in tune)) == set(VOC_CLASSES)
    assert not duplicate_images(
        ((i, dataset.images[i]) for i in tune),
        (dataset.images[i] for i in load_indices("voc_10mb.json")),
    )

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

        assert len(counts) == len(VOC_CLASSES)
        total_bytes = sum(Path(dataset.images[i]).stat().st_size for i in indices)
        assert total_bytes <= SIZE_BYTES[size]

        max_difference = max(
            abs(
                counts[c] / len(indices)
                - original_counts[c] / len(dataset)
            )
            for c in VOC_CLASSES
        )

        print(
            f"{size}: {len(indices)} images, "
            f"{len(VOC_CLASSES)} classes, {total_bytes:,}/{SIZE_BYTES[size]:,} JPEG bytes, "
            f"max class frequency difference: "
            f"{max_difference:.4f}"
        )

        previous = indices

    print("Pascal VOC validation passed.")


if __name__ == "__main__":
    validate_caltech()
    validate_voc()
