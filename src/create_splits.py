
import json
import random
from pathlib import Path
from collections import defaultdict, Counter

from torchvision.datasets import Caltech256

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLIT_DIR = Path(__file__).resolve().parent.parent / "splits"

SEED = 42
TRAIN_RATIO = 0.8


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


if __name__ == "__main__":
    create_caltech_split()
