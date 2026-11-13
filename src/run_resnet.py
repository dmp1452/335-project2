'''
IMPORTANT: SOURCE TO ADD TO COLAB Torchvision contributors. “ResNet-18.”
  https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
  Used to construct ResNet-18 with `weights=None` and replace its final
  classification layer for our datasets.
'''
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataloaders import get_dataloaders
from src.resnet_model import build_resnet, get_loss_func
from src.resnet_training import train_one_epoch, evaluate


def run_experiment(
        dataset_name,
        size,
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        seed=42,
        initializer=None,
        method="baseline",

):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running {dataset_name}, {size}, {method} on {device}")

    train_loader, eval_loader = get_dataloaders(
        dataset_name=dataset_name,
        size=size,
        batch_size=batch_size,
        num_workers=0,
    )

    model = build_resnet(dataset_name, initializer).to(device)
    loss_func = get_loss_func(dataset_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    generator = torch.Generator().manual_seed(seed)
    train_loader.generator = generator
    train_loader.sampler.generator = generator

    hist = []
    cumulative_train_time = 0.0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, loss_func, optimizer, device)
        eval_metrics = evaluate(
            model, eval_loader, loss_func, device, dataset_name)

        cumulative_train_time += train_metrics["train_time"]

        row = {
            "epoch": epoch,
            **train_metrics,
            **eval_metrics,
            "cumulative_train_time": cumulative_train_time
        }
        hist.append(row)

        metric_name = "accuracy" if dataset_name == "caltech" else "micro_f1"

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {row['train_loss']:.4f} | "
            f"Eval loss: {row['eval_loss']:.4f} | "
            f"{metric_name}: {row[metric_name]:.4f} | "
            f"Train time: {row['train_time']:.1f}s"
        )

    plot_hist(hist, dataset_name, size, method)

    return hist


def plot_hist(hist, dataset_name, size, method):
    epochs = [row["epoch"] for row in hist]

    metric_name = ("accuracy" if dataset_name == "caltech" else "micro_f1")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [row["train_loss"]
                          for row in hist], label="Training", marker="o")

    axes[0].plot(epochs, [row["eval_loss"]
                          for row in hist], label="Evaluation", marker="o")
    axes[0].set_title("Loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [row[metric_name] for row in hist], marker="o")
    axes[1].set_title(f"Evaluation {metric_name}")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(bottom=0)

    axes[2].plot(epochs, [row["cumulative_train_time"]
                 for row in hist], marker="o")
    axes[2].set_title("Cumulative Training Time")
    axes[2].set_ylabel("Time (s)")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
    fig.suptitle(f"Resnet-18 | {dataset_name} | {size} | {method}")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    hist = run_experiment(
        dataset_name="voc",
        size="10mb",
        epochs=10,

    )
