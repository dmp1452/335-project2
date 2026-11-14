"""Run paired initialization experiments and keep epoch-level results."""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from itertools import product
from pathlib import Path

from .dataset_metadata import SIZE_BYTES, caltech_image_path

DATASETS = ("caltech", "voc")
ARCHITECTURES = ("resnet", "vit")
METHODS = ("native", "orthogonal")
OPTIMIZERS = {
    "resnet": {"name": "Adam", "lr": 1e-3, "weight_decay": 0.0, "batch_size": 32},
    "vit": {"name": "AdamW", "lr": 1e-4, "weight_decay": 0.01, "batch_size": 8},
}


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dataset_info(loader, dataset_name):
    from torch.utils.data import Subset

    indices = range(len(loader.dataset))
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        indices, dataset = dataset.indices, dataset.dataset
    if dataset_name == "caltech":
        paths = [caltech_image_path(dataset, i) for i in indices]
    else:
        paths = [Path(dataset.dataset.images[i]) for i in indices]

    # Relative image IDs and order identify the manifests across runtimes.
    image_ids = ["/".join(path.parts[-2:]) for path in paths]
    return {
        "images": len(paths),
        "jpeg_bytes": sum(path.stat().st_size for path in paths),
        "ids_sha256": hashlib.sha256("\n".join(image_ids).encode()).hexdigest(),
    }


def run(config, output, *, hardware, resume=False, checkpoint_every=1, keep_checkpoint=False):
    import torch

    from .dataloaders import get_dataloaders
    from .resnet_model import build_resnet, get_loss_func
    from .resnet_training import evaluate, synchronize_device, train_one_epoch
    from .weight_initialization import make_initializer

    start = time.perf_counter()
    seed_everything(config["seed"])
    torch.use_deterministic_algorithms(config["deterministic"])
    torch.backends.cudnn.benchmark = False
    device = torch.device(config["device"])

    architecture = config["architecture"]
    dataset = config["dataset"]
    tuning = config["stage"] == "tune"
    train_loader, eval_loader = get_dataloaders(
        dataset, config["size"], config["batch_size"], config["workers"],
        seed=config["seed"], tuning=tuning, pin_memory=device.type == "cuda",
    )
    info = {
        "train": dataset_info(train_loader, dataset),
        "eval": dataset_info(eval_loader, dataset),
        "target_train_bytes": SIZE_BYTES[config["size"]],
        "eval_split": ("training holdout" if tuning else
                       "Caltech fixed test split" if dataset == "caltech" else "VOC official val"),
    }
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    checkpoint_path = output / "checkpoint.pt"
    previous = None
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous["config"] != config or previous["data"] != info:
            raise ValueError(f"Configuration or dataset changed in {output}")
        if previous["completed"]:
            if not keep_checkpoint:
                checkpoint_path.unlink(missing_ok=True)
            print(f"Already complete: {output}", flush=True)
            return previous
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint in {output}. Use a fresh --out directory.")
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(f"Partial run in {output}. Pass --resume.")

    state = None
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if state["result"]["config"] != config or state["result"]["data"] != info:
            raise ValueError(f"Checkpoint configuration or dataset changed in {output}")

    if architecture == "vit":
        from vit.model import create_model

        build_model = create_model
    else:
        build_model = build_resnet

    synchronize_device(device)
    init_start = time.perf_counter()
    initializer = None
    if state is None and config["method"] == "orthogonal":
        initializer = make_initializer(
            architecture, config["method"], seed=config["seed"], gain=config["gain"]
        )
    model = build_model(dataset, initializer)
    if state:
        model.load_state_dict(state["model"])
    model.to(device)
    if config["vit_grad_checkpoint"]:
        model.set_grad_checkpointing()
    synchronize_device(device)
    init_time = time.perf_counter() - init_start

    loss_func = get_loss_func(dataset)
    optimizer = getattr(torch.optim, config["optimizer"])(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"],
        betas=tuple(config["betas"]), eps=config["eps"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        result = state["result"]
        del state
    else:
        result = {
            "config": config, "data": info, "model_setup_seconds": init_time,
            "setup_hardware": hardware, "total_seconds": 0.0,
            "completed": False, "epochs": [],
        }

    history = result["epochs"]
    if previous is not None and len(previous["epochs"]) == len(history):
        result["total_seconds"] = max(result["total_seconds"], previous["total_seconds"])
    elapsed_before_restart = result["total_seconds"]
    cumulative = history[-1]["cumulative_train_time"] if history else 0.0
    metric = "accuracy" if dataset == "caltech" else "micro_f1"

    print(f"{output.name}: {info['train']['images']:,} train images, "
          f"{info['train']['jpeg_bytes']:,}/{info['target_train_bytes']:,} JPEG bytes", flush=True)
    for epoch in range(len(history) + 1, config["epochs"] + 1):
        # Reconstruct shuffling and model RNG at epoch boundaries after resuming.
        epoch_seed = (config["seed"] + epoch) % 2**32
        seed_everything(epoch_seed)
        train_loader.generator.manual_seed(epoch_seed)

        train = train_one_epoch(model, train_loader, loss_func, optimizer, device, scaler=scaler)
        evaluation = evaluate(
            model, eval_loader, loss_func, device, dataset, amp=config["amp"], compute_map=True
        )
        cumulative += train["train_time"]
        row = {"epoch": epoch, **train, **evaluation, "cumulative_train_time": cumulative}
        if not all(math.isfinite(value) for value in row.values()):
            raise FloatingPointError(f"Non-finite metrics at epoch {epoch} in {output}")

        row["hardware"] = hardware
        history.append(row)
        result["total_seconds"] = elapsed_before_restart + time.perf_counter() - start
        final_epoch = epoch == config["epochs"]
        if epoch == 1 or epoch % checkpoint_every == 0 or final_epoch:
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "result": result,
            }, temporary)
            temporary.replace(checkpoint_path)
        result["total_seconds"] = elapsed_before_restart + time.perf_counter() - start
        atomic_json(result_path, result)
        print(
            f"Epoch {epoch}/{config['epochs']} | train loss {row['train_loss']:.4f} | "
            f"eval loss {row['eval_loss']:.4f} | {metric} {row[metric]:.4f} | "
            f"train {row['train_time']:.1f}s",
            flush=True,
        )

    result["total_seconds"] = elapsed_before_restart + time.perf_counter() - start
    result["completed"] = True
    atomic_json(result_path, result)
    if not keep_checkpoint:
        checkpoint_path.unlink(missing_ok=True)
    return result


def combinations(args):
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    architectures = ARCHITECTURES if args.architecture == "all" else (args.architecture,)
    sizes = ("10mb",) if args.command == "tune" else (
        tuple(SIZE_BYTES) if args.size == "all" else (args.size,)
    )
    methods = ("orthogonal",) if args.command == "tune" else (
        METHODS if args.method == "both" else (args.method,)
    )
    for dataset, architecture, size, method in product(datasets, architectures, sizes, methods):
        gains = args.gains if args.command == "tune" else (args.gain if method == "orthogonal" else None,)
        for gain in gains:
            yield dataset, architecture, size, method, gain


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    for command in ("run", "tune"):
        sub = commands.add_parser(command)
        sub.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
        sub.add_argument("--architecture", choices=(*ARCHITECTURES, "all"), default="all")
        sub.add_argument("--epochs", type=int, default=3 if command == "tune" else 10)
        sub.add_argument("--batch-size", type=int)
        sub.add_argument("--lr", type=float)
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--workers", type=int, default=2)
        sub.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
        sub.add_argument("--no-amp", action="store_true")
        sub.add_argument("--deterministic", action="store_true")
        sub.add_argument("--vit-grad-checkpoint", action="store_true")
        sub.add_argument("--checkpoint-every", type=int, default=1)
        sub.add_argument("--resume", action="store_true")
        sub.add_argument("--keep-checkpoint", action="store_true")
        sub.add_argument("--out", type=Path, default=Path("results"))
        sub.add_argument("--dry-run", action="store_true", help="List configurations without loading data or models")
        if command == "run":
            sub.add_argument("--size", choices=(*SIZE_BYTES, "all"), default="all")
            sub.add_argument("--method", choices=(*METHODS, "both"), default="both")
            gains = sub.add_mutually_exclusive_group()
            gains.add_argument("--gain", type=float, default=1.0)
            gains.add_argument("--tuning-results", type=Path, help="Path to a completed tuning_summary.json")
        else:
            sub.add_argument("--gains", type=float, nargs="+", default=[0.7, 1.0, 1.4])
    return cli


def main():
    cli = parser()
    args = cli.parse_args()
    if (args.epochs < 1 or args.workers < 0 or args.checkpoint_every < 1
            or (args.batch_size is not None and args.batch_size < 1) or not 0 <= args.seed < 2**32
            or (args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0))):
        cli.error("Invalid epochs, workers, checkpoint interval, batch size, seed, or learning rate")
    gains = args.gains if args.command == "tune" else [args.gain]
    if any(not math.isfinite(gain) or gain <= 0 for gain in gains) or len(gains) != len(set(gains)):
        cli.error("Gains must be unique, finite, and positive")

    selections = {}
    if args.command == "run" and args.tuning_results:
        records = json.loads(args.tuning_results.read_text())["selected"]
        selections = {(row["dataset"], row["architecture"]): row["gain"] for row in records}
    jobs = []
    for dataset, architecture, size, method, gain in combinations(args):
        if args.command == "run" and args.tuning_results and method == "orthogonal":
            if (dataset, architecture) not in selections:
                cli.error(f"Missing tuned gain for {dataset}/{architecture}")
            gain = selections[dataset, architecture]
            if not math.isfinite(gain) or gain <= 0:
                cli.error(f"Invalid tuned gain for {dataset}/{architecture}")
        jobs.append((dataset, architecture, size, method, gain))
    if args.dry_run:
        for dataset, arch, size, method, gain in jobs:
            print(f"{dataset} {arch} {size} {method}" + (f" gain={gain:g}" if gain else ""))
        print(f"{len(jobs)} runs")
        return

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from filelock import FileLock
    from importlib.metadata import version

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        cli.error("CUDA is unavailable. Enable a GPU runtime or pass --device cpu.")
    versions = {name: version(name) for name in ("torch", "torchvision", "timm", "scikit-learn")}
    hardware = torch.cuda.get_device_name(0) if device == "cuda" else platform.machine()
    stage_dir = args.out / args.command
    trials = []
    for dataset, architecture, size, method, gain in jobs:
        defaults = OPTIMIZERS[architecture]
        config = {
            "stage": args.command, "dataset": dataset, "architecture": architecture, "size": size,
            "method": method, "gain": gain, "seed": args.seed, "epochs": args.epochs,
            "batch_size": args.batch_size or defaults["batch_size"], "workers": args.workers,
            "optimizer": defaults["name"], "lr": args.lr or defaults["lr"],
            "weight_decay": defaults["weight_decay"], "betas": [0.9, 0.999], "eps": 1e-8,
            "amp": device == "cuda" and not args.no_amp, "device": device,
            "cuda_version": torch.version.cuda, "versions": versions,
            "deterministic": args.deterministic,
            "vit_grad_checkpoint": architecture == "vit" and args.vit_grad_checkpoint,
        }
        run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        path = stage_dir / f"{dataset}-{architecture}-{size}-{method}-{run_id}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock", timeout=0):
            result = run(
                config, path, hardware=hardware, resume=args.resume, checkpoint_every=args.checkpoint_every,
                keep_checkpoint=args.keep_checkpoint,
            )
        if args.command == "tune":
            metric = "accuracy" if dataset == "caltech" else "micro_f1"
            history = result["epochs"]
            trials.append({
                "dataset": dataset, "architecture": architecture, "gain": gain,
                "metric": metric, "mean_score": sum(row[metric] for row in history) / len(history),
                "final_eval_loss": history[-1]["eval_loss"], "result": str(path.relative_to(stage_dir)),
            })
    if trials:
        trial_groups = {(t["dataset"], t["architecture"]) for t in trials}
        selected = []
        for dataset, architecture in sorted(trial_groups):
            candidates = [t for t in trials if (t["dataset"], t["architecture"]) == (dataset, architecture)]
            selected.append(min(candidates, key=lambda t: (-t["mean_score"], t["final_eval_loss"], t["gain"])))
        summary_path = stage_dir / "tuning_summary.json"
        with FileLock(str(summary_path) + ".lock", timeout=10):
            if summary_path.exists():
                prior = json.loads(summary_path.read_text())
                for field, rows in (("trials", trials), ("selected", selected)):
                    rows.extend(
                        row for row in prior[field]
                        if (row["dataset"], row["architecture"]) not in trial_groups
                    )
            atomic_json(summary_path, {
                "criterion": "Highest mean holdout score over epochs, then lowest final loss, then smallest gain",
                "trials": trials, "selected": selected,
            })
        for row in selected:
            print(f"{row['dataset']}/{row['architecture']} selected gain {row['gain']:g}")


if __name__ == "__main__":
    main()
