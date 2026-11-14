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


def train_one_epoch(model, loader, loss_func, optimzer, device, *, scaler=None):
    model.train()

    total_loss = torch.zeros((), device=device)
    total_samples = 0

    synchronize_device(device)
    start_time = time.perf_counter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimzer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type, enabled=scaler is not None and scaler.is_enabled()
        ):
            outputs = model(images)
            loss = loss_func(outputs, labels)

        if scaler is None:
            loss.backward()
            optimzer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimzer)
            scaler.update()

        batch_size = images.size(0)
        total_loss += loss.detach().float() * batch_size
        total_samples += batch_size

    synchronize_device(device)
    elapsed_secs = time.perf_counter() - start_time

    return {
        "train_loss": total_loss.item() / total_samples,
        "train_time": elapsed_secs
    }


@torch.no_grad()
def evaluate(model, loader, loss_func, device, dataset_name, *, amp=False, compute_map=False):
    if dataset_name not in {"caltech", "voc"}:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    model.eval()

    total_loss = 0.0
    total_samples = 0

    correct = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    all_scores, all_labels = [], []

    synchronize_device(device)
    start_time = time.perf_counter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp):
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
            if compute_map:
                all_scores.append(outputs.float().cpu())
                all_labels.append(targets.cpu())

            true_positives += (predictions & targets).sum().item()
            false_positives += (predictions & ~targets).sum().item()
            false_negatives += (~predictions & targets).sum().item()

    metrics = {"eval_loss": total_loss / total_samples}

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

        if compute_map:
            from sklearn.metrics import average_precision_score

            # AP uses the ordering of logits. Sigmoid is monotone.
            metrics["mAP"] = float(average_precision_score(
                torch.cat(all_labels).numpy(), torch.cat(all_scores).numpy(), average="macro"
            ))

    synchronize_device(device)
    metrics["eval_time"] = time.perf_counter() - start_time
    return metrics
