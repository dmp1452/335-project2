"""Plot completed primary runs and gain trials from experiment JSON files."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .experiment import ARCHITECTURES, DATASETS, METHODS, SIZE_BYTES

COLORS = {"native": "#285f94", "orthogonal": "#ab4620"}
ARCH_NAMES = {"resnet": "ResNet-18", "vit": "ViT-Base"}


def primary_plots(source, destination):
    grouped = defaultdict(dict)
    for path in sorted((source / "run").glob("*/results.json")):
        result = json.loads(path.read_text())
        if not result["completed"]:
            continue
        config = result["config"]
        # Match both methods by run configuration and the actual data manifests.
        comparable = {key: value for key, value in config.items() if key not in {"method", "gain"}}
        group_id = json.dumps([comparable, result["data"]], sort_keys=True)
        grouped[group_id][config["method"], config["gain"]] = result

    by_case = defaultdict(lambda: defaultdict(list))
    for group_id, runs in grouped.items():
        config = next(iter(runs.values()))["config"]
        by_case[config["dataset"], config["size"], config["seed"]][config["architecture"]].append(
            (group_id, runs)
        )

    for (dataset, size, seed), by_arch in sorted(by_case.items()):
        # If several configurations exist, keep their curves in separate figures.
        count = max(map(len, by_arch.values()))
        for index in range(count):
            cases = {
                arch: entries[index] for arch, entries in by_arch.items() if index < len(entries)
            }
            score = "accuracy" if dataset == "caltech" else "micro_f1"
            labels = [
                ("epoch", "Loss", "Epoch"),
                ("epoch", "Accuracy" if dataset == "caltech" else "Micro F1", "Epoch"),
                ("cumulative_train_time", "Accuracy" if dataset == "caltech" else "Micro F1",
                 "Cumulative training (s)"),
            ]
            if dataset == "voc":
                labels.append(("cumulative_train_time", "Macro mAP", "Cumulative training (s)"))
            fig, axes = plt.subplots(2, len(labels), figsize=(4.15 * len(labels), 7), squeeze=False)
            for arch, row in zip(ARCHITECTURES, axes):
                for ax, (_, ylabel, xlabel) in zip(row, labels):
                    ax.set(xlabel=xlabel, ylabel=ylabel)
                    ax.grid(alpha=0.2)
                if arch not in cases:
                    row[0].set_title(f"{ARCH_NAMES[arch]} · pending")
                    for ax in row:
                        ax.set_axis_off()
                    continue
                _, runs = cases[arch]
                missing = set(METHODS) - {method for method, _ in runs}
                title = ARCH_NAMES[arch] + (" · awaiting " + ", ".join(sorted(missing)) if missing else "")
                row[0].set_title(title)
                profiles = {tuple(sorted({item["hardware"] for item in result["epochs"]}))
                            for result in runs.values()}
                comparable_time = len(profiles) == 1 and len(next(iter(profiles))) == 1
                for (method, gain), result in sorted(runs.items()):
                    history = result["epochs"]
                    name = "Native" if method == "native" else f"Orthogonal (gain {gain:g})"
                    color = COLORS[method]
                    epochs = [item["epoch"] for item in history]
                    times = [item["cumulative_train_time"] for item in history]
                    scores = [item[score] for item in history]
                    row[0].plot(epochs, [item["train_loss"] for item in history], "--",
                                color=color, label=f"{name} train")
                    row[0].plot(epochs, [item["eval_loss"] for item in history],
                                color=color, marker=".", label=f"{name} eval")
                    row[1].plot(epochs, scores, color=color, marker=".", label=name)
                    if comparable_time:
                        row[2].plot(times, scores, color=color, marker=".", label=name)
                        if dataset == "voc":
                            row[3].plot(times, [item["mAP"] for item in history],
                                        color=color, marker=".", label=name)
                if not comparable_time:
                    for ax in row[2:]:
                        ax.set_title("Training used different hardware")
                for ax in row:
                    if ax.lines:
                        ax.legend(fontsize=8)
            datasets = {"caltech": "Caltech-256", "voc": "PASCAL VOC 2012"}
            subtitle = []
            for arch, (_, runs) in cases.items():
                result = next(iter(runs.values()))
                cfg, data = result["config"], result["data"]
                mb = data["train"]["jpeg_bytes"] / 1_000_000
                hardware = sorted({item["hardware"] for run in runs.values() for item in run["epochs"]})
                subtitle.append(
                    f"{ARCH_NAMES[arch]} {mb:.2f} MB, batch {cfg['batch_size']}, "
                    f"{'/'.join(hardware)}, LR {cfg['lr']:g}"
                )
            fig.suptitle(f"{datasets[dataset]} · {size.upper()} · seed {seed}\n" + " | ".join(subtitle))
            fig.tight_layout()
            digest = hashlib.sha256("".join(sorted(group for group, _ in cases.values())).encode()).hexdigest()[:8]
            fig.savefig(destination / f"{dataset}-{size}-seed{seed}-{digest}.png", dpi=145)
            plt.close(fig)
    if not by_case:
        print("No completed primary runs found.")


def tuning_plot(source, destination):
    path = source / "tune" / "tuning_summary.json"
    if not path.exists():
        print("No completed tuning summary found.")
        return
    summary = json.loads(path.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), squeeze=False)
    for dataset, row in zip(DATASETS, axes):
        for arch, ax in zip(ARCHITECTURES, row):
            trials = sorted(
                (t for t in summary["trials"] if (t["dataset"], t["architecture"]) == (dataset, arch)),
                key=lambda t: t["gain"],
            )
            ax.set_title(f"{dataset.upper()} · {ARCH_NAMES[arch]}")
            if not trials:
                ax.set_axis_off()
                continue
            winner = next(t for t in summary["selected"] if (t["dataset"], t["architecture"]) == (dataset, arch))
            ax.plot([t["gain"] for t in trials], [t["mean_score"] for t in trials], "o-", color="#285f94")
            ax.scatter(winner["gain"], winner["mean_score"], color="#ab4620", label="Selected")
            ax.set(xlabel="Orthogonal gain", ylabel=f"Mean holdout {winner['metric'].replace('_', ' ')}")
            ax.set_xticks([t["gain"] for t in trials])
            ax.grid(alpha=0.2)
            ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "gain-tuning.png", dpi=145)
    plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--results", type=Path, default=Path("results"))
    cli.add_argument("--out", type=Path, default=Path("plots"))
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    primary_plots(args.results, args.out)
    tuning_plot(args.results, args.out)
