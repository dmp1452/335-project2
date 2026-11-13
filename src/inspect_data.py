
from pathlib import Path
from collections import Counter
from torchvision.datasets import Caltech256, VOCDetection

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"
]


def inspect_caltech():
    dataset = Caltech256(
        root=str(DATA_DIR),
        download=False
    )

    counts = Counter(dataset.y)

    print("\n=== Caltech-256 ===")
    print(f"Total images: {len(dataset)}")
    print(f"Total classes: {len(counts)}")

    for class_id, count in sorted(counts.items()):
        print(f"Class {class_id}: {count} images")


def inspect_voc():
    for split in ["train", "val"]:
        dataset = VOCDetection(
            root=str(DATA_DIR),
            year="2012",
            image_set=split,
            download=False
        )

        counts = Counter()

        for _, annotation in dataset:
            objects = annotation["annotation"].get("object", [])

            if isinstance(objects, dict):
                objects = [objects]

            # Count images containing each class
            labels = {obj["name"] for obj in objects}

            counts.update(labels)

        print(f"\n=== Pascal VOC 2012 ({split}) ===")
        print(f"Total images: {len(dataset)}")

        for class_name in VOC_CLASSES:
            print(f"{class_name}: {counts[class_name]} images")


if __name__ == "__main__":
    inspect_caltech()
    inspect_voc()
