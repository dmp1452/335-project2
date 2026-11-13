import time
import torch


def synchronize_device(device):
    """
    Synchronizes the device to ensure that all operations are completed before proceeding.

    Args:
        device: The device to synchronize (e.g., 'cuda' or 'cpu').
    """
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def train_one_epoch(model, loader, loss_func, optimzer, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    synchronize_device(device)
    start_time = time.perf_counter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimzer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = loss_func(outputs, labels)

        loss.backward()
        optimzer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    synchronize_device(device)
    elapsed_secs = time.perf_counter() - start_time

    return {
        "train_loss": total_loss / total_samples,
        "train_time": elapsed_secs
    }


@torch.no_grad()
def evaluate(model, loader, loss_func, device, dataset_name):
    if dataset_name not in {"caltech", "voc"}:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    model.eval()

    total_loss = 0.0
    total_samples = 0

    correct = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    synchronize_device(device)
    start_time = time.perf_counter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_func(outputs, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if dataset_name == "caltech":
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()

        else:
            predictions = outputs >= 0
            targets = labels.bool()

            true_positives += (predictions & targets).sum().item()
            false_positives += (predictions & ~targets).sum().item()
            false_negatives += (~predictions & targets).sum().item()

    synchronize_device(device)
    elapsed_secs = time.perf_counter() - start_time

    metrics = {
        "eval_loss": total_loss / total_samples,
        "eval_time": elapsed_secs
    }

    if dataset_name == "caltech":
        metrics["accuracy"] = correct / total_samples

    else:
        precision_denom = true_positives + false_positives
        recall_denom = true_positives + false_negatives

        f1_denom = 2 * true_positives + false_positives + false_negatives

        metrics["micro_precision"] = true_positives / \
            precision_denom if precision_denom > 0 else 0.0
        metrics["micro_recall"] = true_positives / \
            recall_denom if recall_denom > 0 else 0.0
        metrics["micro_f1"] = (2 * true_positives) / \
            f1_denom if f1_denom > 0 else 0.0

    return metrics
