# 335-project2
Member	Main responsibility	Specific tasks


Danny 1: Dataset preparation	Download and preprocess the images	Prepare Pascal VOC and Caltech-256. Create the three sizes for each dataset while preserving classes and class balance. Build shared data loaders and keep evaluation data consistent across experiments. Document preprocessing.


Matthew 2: ResNet experiments	Implement and train ResNet	Set up a ResNet model, run its baseline initialization, then run the group’s proposed initialization. Test all six dataset versions and save training time, loss, and evaluation metrics per epoch.



Shannon 3: ViT experiments	Implement and train ViT-base	Set up ViT-base and run the same baseline-versus-proposed-initialization experiments across all six dataset versions. Save results in the same format as Member 2.


Asdaq 4: Weight initialization method	Research and implement the speedup approach	Find at least three related papers, explain the chosen method, and implement initialization code for both architectures. Coordinate with Members 2 and 3 and lead at least one tuning experiment, such as testing initialization scale.


Yujin 5: Evaluation and integration	Compare results and assemble the final submission	Define shared logging and evaluation functions early. Produce baseline-versus-method plots, compare training speed and predictive performance, organize the final Colab notebook, and verify that it runs from start to finish. Coordinate slides and recording.

# Project 2 - Dataset Preparation

Dataset preparation for Caltech-256 and Pascal VOC 2012, used for training ResNet and ViT models.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Download both datasets:

```bash
python src/download_data.py
```

The datasets are stored in `data/`, which is excluded from Git.

## Preprocessing

Images are processed using `src/preprocessing.py`:

- Converted to RGB
- Resized to 224×224
- Converted to PyTorch tensors
- Normalized using ImageNet mean and standard deviation

Preprocessing is applied when images are loaded, so the original files are unchanged.

## Dataset Splits

Generate the training subsets:

```bash
python src/create_splits.py
```

This creates three versions of each dataset: 10 MB, 100 MB, and 1 GB (or the full available training set).

- **Caltech-256:** 80/20 stratified training/test split, with 257 classes.
- **Pascal VOC:** Uses the official training and validation sets, with 20 multi-label classes.

Subset indices are saved in `splits/`. All subsets preserve class coverage and approximate class distributions.

The largest Caltech and VOC training subsets are approximately 919 MB and 665 MB, respectively.

## Using the DataLoaders

```python
from src.dataloaders import get_dataloaders

train_loader, val_loader = get_dataloaders(
    dataset_name="caltech",
    size="100mb",
    batch_size=32
)
```

Options:
- `dataset_name`: `"caltech"` or `"voc"`
- `size`: `"10mb"`, `"100mb"`, or `"1gb"`
- `batch_size`: Defaults to 32
- `num_workers`: Defaults to 0

Both datasets return images with shape `[batch_size, 3, 224, 224]`.

Caltech returns integer class labels (0–256). VOC returns 20-element multi-label vectors.

The evaluation set stays the same across all three training sizes.

## Validation

Check class coverage, distribution, and split consistency:

```bash
python -m src.validate_data
```

Test the DataLoaders:

```bash
python -m src.test_dataloaders
```

All commands should be run from the repository root.
