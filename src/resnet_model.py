from torch import nn
from torchvision.models import resnet18


def build_resnet(dataset_name, initializer=None):
    """
    Builds a ResNet-18 model for the given dataset.

    Args:
        dataset_name: The name of the dataset for which the model is being built.
        initializer: Optional initializer for the model's weights.

    Returns:
        A ResNet-18 model initialized for the given dataset.
    """
    if dataset_name == "caltech":
        num_classes = 257
    elif dataset_name == "voc":
        num_classes = 20
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    model = resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if initializer is not None:
        # whoever is doing part 4 will implement the initializer function. This is a placeholder
        initializer(model)

    return model


def get_loss_func(dataset_name):
    """
    Returns the appropriate loss function for the given dataset.

    Args:
        dataset_name: The name of the dataset for which the loss function is being retrieved.

    Returns:
        The appropriate loss function for the given dataset.
    """
    if dataset_name == "caltech":
        return nn.CrossEntropyLoss()

    if dataset_name == "voc":
        return nn.BCEWithLogitsLoss()

    raise ValueError(f"Unsupported dataset: {dataset_name}")
