import torch
import time
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def train_model(dataset_name, size, model, train_loader, val_loader, epochs = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_losses = []
    val_losses = []
    performance_scores = []
    epoch_times = []

    if dataset_name == "caltech":
        criterion = torch.nn.CrossEntropyLoss()
    elif dataset_name == "voc":
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Unknown dataset")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 1e-4
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled = (device.type == device.type)
    )

    for epoch in range(epochs):
        start = time.perf_counter()
        model.train()

        # Train loop
        train_start = time.perf_counter()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            if dataset_name == "caltech":
                labels = labels.long()
            else:
                labels = labels.float()
            optimizer.zero_grad()
            with torch.autocast(
                device_type = "cuda",
                dtype = torch.float16,
                enabled = (device.type == "cuda")
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels.size(0)
            if dataset_name == "caltech":
                predictions = outputs.argmax(1)
                correct += (predictions == labels).sum().item()
            else:
                predictions = (outputs >= 0).float()
                correct += (predictions == labels).sum().item()

            if dataset_name == "caltech":
                total += labels.size(0)
            else:
                total += labels.numel()
        train_time = time.perf_counter() - train_start
        train_loss = total_loss / len(train_loader.dataset)
        if dataset_name == "caltech":
            train_acc = correct / total

        # Validation loop
        val_start = time.perf_counter()
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                if dataset_name == "caltech":
                    labels = labels.long()
                else:
                    labels = labels.float()
                with torch.autocast(
                    device_type = device.type,
                    dtype = torch.float16,
                    enabled = (device.type == "cuda")
                ):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                if dataset_name == "caltech":
                    predictions = outputs.argmax(1)
                    val_correct += (predictions == labels).sum().item()
                else:
                    predictions = (outputs >= 0).int()
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                if dataset_name == "caltech":
                    val_total += labels.size(0)
            val_time = time.perf_counter() - val_start

            if dataset_name == "caltech":
                val_acc = val_correct / val_total

            else:
                val_precision = precision_score(
                    all_labels, all_predictions, average = "micro", zero_division = 0
                )

                val_recall = recall_score(
                    all_labels, all_predictions, average = "micro", zero_division = 0
                )

                val_f1 = f1_score(
                    all_labels, all_predictions, average = "micro", zero_division = 0
                )

            val_loss = val_loss / len(val_loader.dataset)

        epoch_time = time.perf_counter() - start

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_times.append(epoch_time)

        if dataset_name == "caltech":
            performance_scores.append(val_acc)
        else:
            performance_scores.append(val_f1)

        print(
            f"Epoch: [{epoch + 1}/{epochs}] "
            f"Training Loss: {train_loss:.4f} "
            f"Validation Loss: {val_loss:.4f} "
        )

        if dataset_name == "caltech":
            print(
                f"Training Accuracy: {train_acc:.4f} "
                f"Validation Accuracy: {val_acc:.4f}"
            )
        else:
            print(
                f"Micro Precision: {val_precision:.4f} "
                f"Micro Recall: {val_recall:.4f} "
                f"Micro F1: {val_f1:.4f}"
            )

        print(f"Total Epoch Time: {epoch_time:.2f} seconds")

    epochs_range = range(1, epochs + 1)

    # Training Loss
    plt.figure()
    plt.plot(epochs_range, train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(f"{dataset_name}_{size}_training_loss.png")
    plt.close()

    # Validation Loss
    plt.figure()
    plt.plot(epochs_range, val_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.savefig(f"{dataset_name}_{size}_validation_loss.png")
    plt.close()

    # Accuracy / Micro F1
    plt.figure()
    plt.plot(epochs_range, performance_scores)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy" if dataset_name == "caltech" else "Micro F1")
    plt.title("Model Performance")
    plt.savefig(f"{dataset_name}_{size}_model_performance.png")
    plt.close()

    # Training Time
    plt.figure()
    plt.plot(epochs_range, epoch_times)
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.title("Epoch Time")
    plt.savefig(f"{dataset_name}_{size}_training_time.png")
    plt.close()

    return model