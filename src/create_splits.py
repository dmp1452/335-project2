
import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from torchvision.datasets import VOCDetection

import numpy as np

from torchvision.datasets import Caltech256

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLIT_DIR = Path(__file__).resolve().parent.parent / "splits"

SEED = 42
TRAIN_RATIO = 0.8

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"
]



def create_voc_subsets():
    dataset = VOCDetection(
        root=str(DATA_DIR),
        year="2012",
        image_set="train",
        download=False
    )

    rng = random.Random(SEED)

    image_labels = {}
    image_sizes = {}

    # Read image labels and file sizes
    for index in range(len(dataset)):
        _, annotation = dataset[index]

        objects = annotation["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]

        image_labels[index] = {
            obj["name"] for obj in objects
            if obj["name"] in VOC_CLASSES
        }

        image_sizes[index] = Path(dataset.images[index]).stat().st_size

    # Calculate original class frequencies
    original_counts = Counter()

    for labels in image_labels.values():
        original_counts.update(labels)

    original_freq = {
        c: original_counts[c] / len(dataset)
        for c in VOC_CLASSES
    }

    def select_subset(target_bytes, required_indices=None, trials=300):
        """
        Randomly sample candidate subsets and select the one
        whose class frequencies most closely match the original.
        """
        required_indices = set(required_indices or [])
        all_indices = set(range(len(dataset)))
        remaining = sorted(all_indices - required_indices)

        best_subset = None
        best_score = float("inf")

        for _ in range(trials):
            candidates = remaining.copy()
            rng.shuffle(candidates)

            selected = list(sorted(required_indices))
            total_bytes = sum(image_sizes[i] for i in selected)

            for index in candidates:
                size = image_sizes[index]

                if total_bytes + size <= target_bytes:
                    selected.append(index)
                    total_bytes += size

            counts = Counter()
            for index in selected:
                counts.update(image_labels[index])

            # Reject subsets missing any class
            if len(counts) != len(VOC_CLASSES):
                continue

            # Minimize the maximum class-frequency difference
            score = max(
                abs(counts[c] / len(selected) - original_freq[c])
                for c in VOC_CLASSES
            )

            if score < best_score:
                best_score = score
                best_subset = selected

        if best_subset is None:
            raise RuntimeError(
                "Could not find a subset containing all classes."
            )

        return sorted(best_subset)

    # Select 10 MB, then expand it to 100 MB
    # This ensures the subsets are nested
    subset_10mb = select_subset(10_000_000, trials=1000)

    subset_100mb = select_subset(
        100_000_000,
        required_indices=subset_10mb,
        trials=500
    )

    # Use the entire training set for the largest version
    subset_1gb = list(range(len(dataset)))

    subsets = {
        "10mb": subset_10mb,
        "100mb": subset_100mb,
        "1gb": subset_1gb
    }

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for name, indices in subsets.items():
        with open(SPLIT_DIR / f"voc_{name}.json", "w") as f:
            json.dump(indices, f, indent=2)

        counts = Counter()
        for index in indices:
            counts.update(image_labels[index])

        total_bytes = sum(image_sizes[i] for i in indices)

        max_difference = max(
            abs(counts[c] / len(indices) - original_freq[c])
            for c in VOC_CLASSES
        )

        print(f"\n=== Pascal VOC {name} ===")
        print(f"Images: {len(indices)}")
        print(f"Size: {total_bytes / 1_000_000:.2f} MB")
        print(f"Classes represented: {len(counts)}/20")
        print(f"Max class frequency difference: {max_difference:.4f}")

    print("\nVOC validation set remains unchanged.")




def create_caltech_split():
    dataset = Caltech256(
        root=str(DATA_DIR),
        download=False
    )

    # Group image indices by class
    class_indices = defaultdict(list)

    for index, label in enumerate(dataset.y):
        class_indices[label].append(index)

    rng = random.Random(SEED)

    train_indices = []
    test_indices = []

    # Split each class independently
    for label, indices in sorted(class_indices.items()):
        rng.shuffle(indices)

        split_point = int(len(indices) * TRAIN_RATIO)

        train_indices.extend(indices[:split_point])
        test_indices.extend(indices[split_point:])

    # Save reproducible splits
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SPLIT_DIR / "caltech_train.json", "w") as f:
        json.dump(sorted(train_indices), f, indent=2)

    with open(SPLIT_DIR / "caltech_test.json", "w") as f:
        json.dump(sorted(test_indices), f, indent=2)

    # Verify results
    train_counts = Counter(dataset.y[i] for i in train_indices)
    test_counts = Counter(dataset.y[i] for i in test_indices)

    print("=== Caltech-256 Split ===")
    print(f"Total images: {len(dataset)}")
    print(f"Training images: {len(train_indices)}")
    print(f"Evaluation images: {len(test_indices)}")
    print(f"Classes in training: {len(train_counts)}")
    print(f"Classes in evaluation: {len(test_counts)}")

    assert set(train_indices).isdisjoint(test_indices)
    assert len(train_indices) + len(test_indices) == len(dataset)
    assert len(train_counts) == len(test_counts) == 257

    print("All split checks passed.")


def inspect_caltech_sizes():
    dataset = Caltech256(
        root=str(DATA_DIR),
        download=False
    )

    with open(SPLIT_DIR / "caltech_train.json") as f:
        train_indices = json.load(f)

    total_bytes = 0
    smallest_per_class = {}

    for index in train_indices:
        label = dataset.y[index]

        image_path = (
            Path(dataset.root)
            / "256_ObjectCategories"
            / dataset.categories[label]
            / f"{label + 1:03d}_{dataset.index[index]:04d}.jpg"
        )

        size = image_path.stat().st_size
        total_bytes += size

        if label not in smallest_per_class:
            smallest_per_class[label] = size
        else:
            smallest_per_class[label] = min(
                smallest_per_class[label], size
            )

    min_bytes = sum(smallest_per_class.values())

    print("\n=== Caltech-256 Storage ===")
    print(f"Training images: {len(train_indices)}")
    print(f"Total training size: {total_bytes / 1_000_000:.2f} MB")
    print(f"Minimum size covering all classes: {min_bytes / 1_000_000:.2f} MB")

    for target_mb in [10, 100, 1000]:
        target_bytes = target_mb * 1_000_000
        feasible = total_bytes >= target_bytes

        print(
            f"{target_mb} MB target: "
            f"{'Possible by total size' if feasible else 'Exceeds training pool'}"
        )


def create_caltech_subsets():
    dataset = Caltech256(
        root=str(DATA_DIR),
        download=False
    )

    with open(SPLIT_DIR / "caltech_train.json") as f:
        train_indices = json.load(f)

    with open(SPLIT_DIR / "caltech_test.json") as f:
        test_indices = set(json.load(f))

    rng = random.Random(SEED)

    class_indices = defaultdict(list)
    image_sizes = {}

    for index in train_indices:
        label = dataset.y[index]

        image_path = (
            Path(dataset.root)
            / "256_ObjectCategories"
            / dataset.categories[label]
            / f"{label + 1:03d}_{dataset.index[index]:04d}.jpg"
        )

        class_indices[label].append(index)
        image_sizes[index] = image_path.stat().st_size

    for indices in class_indices.values():
        rng.shuffle(indices)

    ordered_indices = []

    positions = {label: 0 for label in class_indices}
    selected_counts = Counter()

    for label in sorted(class_indices):
        index = class_indices[label][0]
        ordered_indices.append(index)

        positions[label] = 1
        selected_counts[label] = 1

    while len(ordered_indices) < len(train_indices):
        available = [
            label for label, indices in class_indices.items()
            if positions[label] < len(indices)
        ]

        label = min(
            available,
            key=lambda c: (
                selected_counts[c] + 1
            ) / len(class_indices[c])
        )

        index = class_indices[label][positions[label]]
        positions[label] += 1

        ordered_indices.append(index)
        selected_counts[label] += 1

    targets = {
        "10mb": 10_000_000,
        "100mb": 100_000_000,
        "1gb": float("inf")
    }

    for name, target_bytes in targets.items():
        selected = []
        total_bytes = 0

        for index in ordered_indices:
            size = image_sizes[index]

            if total_bytes + size > target_bytes:
                if name != "1gb":
                    continue

            selected.append(index)
            total_bytes += size

        selected = sorted(selected)

        with open(SPLIT_DIR / f"caltech_{name}.json", "w") as f:
            json.dump(selected, f, indent=2)

        counts = Counter(dataset.y[i] for i in selected)

        assert not set(selected).intersection(test_indices)

        print(f"\n=== Caltech {name} ===")
        print(f"Images: {len(selected)}")
        print(f"Size: {total_bytes / 1_000_000:.2f} MB")
        print(f"Classes represented: {len(counts)}/257")


if __name__ == "__main__":
    create_caltech_split()
    inspect_caltech_sizes()
    create_caltech_subsets()
    create_voc_subsets()