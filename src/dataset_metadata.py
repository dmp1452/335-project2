"""Image paths and labels shared by split creation and validation."""

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from xml.etree.ElementTree import parse

CALTECH_CLASSES = 256
SIZE_BYTES = {"10mb": 10_000_000, "100mb": 100_000_000, "1gb": 1_000_000_000}
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"
]


def caltech_image_path(dataset, index):
    label = dataset.y[index]
    return (
        Path(dataset.root) / "256_ObjectCategories" / dataset.categories[label]
        / f"{label + 1:03d}_{dataset.index[index]:04d}.jpg"
    )


def voc_labels_from_xml(path):
    return {
        name.text for name in parse(path).getroot().findall("object/name")
        if name.text in VOC_CLASSES
    }


def duplicate_images(candidates, holdout):
    """Return candidate IDs whose full JPEG bytes also occur in the holdout."""
    # Byte-identical files have equal sizes, so hash only potential matches.
    paths = defaultdict(list)
    for path in holdout:
        path = Path(path)
        paths[path.stat().st_size].append(path)
    hashes = {}
    duplicates = set()
    for index, path in candidates:
        path = Path(path)
        size = path.stat().st_size
        if size not in paths:
            continue
        if size not in hashes:
            hashes[size] = {sha256(p.read_bytes()).digest() for p in paths[size]}
        if sha256(path.read_bytes()).digest() in hashes[size]:
            duplicates.add(index)
    return duplicates
