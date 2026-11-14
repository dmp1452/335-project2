
from pathlib import Path
from torchvision.datasets import Caltech256, VOCDetection
from torchvision.datasets.utils import download_and_extract_archive

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_caltech256():
    print("Downloading Caltech-256...")
    root = DATA_DIR / "caltech256"
    if not (root / "256_ObjectCategories").is_dir():
        filename = "256_ObjectCategories.tar"
        try:
            download_and_extract_archive(
                f"https://data.caltech.edu/records/nyy15-4j048/files/{filename}?download=1",
                str(root), filename=filename,
                md5="67b4f42ca05d46448c6bb8ecd2220f6d",
                remove_finished=True,
            )
        except (OSError, RuntimeError):
            download_and_extract_archive(
                "https://huggingface.co/datasets/xingslong/caltech256/resolve/"
                f"6fb15dfa9b9cb4e270668821b6b159e1f39bf141/{filename}",
                str(root), filename=filename,
                md5="67b4f42ca05d46448c6bb8ecd2220f6d",
                remove_finished=True,
            )

    dataset = Caltech256(root=str(DATA_DIR), download=False)
    print(f"Caltech-256: {len(dataset)} images")
    return dataset


def download_pascal_voc():
    print("Downloading Pascal VOC 2012...")
    if not (DATA_DIR / "VOCdevkit" / "VOC2012").is_dir():
        filename = "VOCtrainval_11-May-2012.tar"
        try:
            download_and_extract_archive(
                f"https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/{filename}",
                str(DATA_DIR),
                md5="6cd6e144f989b92b3379bac3b3de84fd",
                remove_finished=True,
            )
        except (OSError, RuntimeError):
            download_and_extract_archive(
                f"https://pjreddie.com/media/files/{filename}",
                str(DATA_DIR),
                md5="6cd6e144f989b92b3379bac3b3de84fd",
                remove_finished=True,
            )

    train = VOCDetection(root=str(DATA_DIR), year="2012", image_set="train")
    val = VOCDetection(root=str(DATA_DIR), year="2012", image_set="val")

    print(f"Pascal VOC training: {len(train)} images")
    print(f"Pascal VOC validation: {len(val)} images")

    return train, val


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description="Download the original datasets")
    cli.add_argument("--dataset", choices=("caltech", "voc", "all"), default="all")
    args = cli.parse_args()
    if args.dataset in {"caltech", "all"}:
        download_caltech256()
    if args.dataset in {"voc", "all"}:
        download_pascal_voc()
