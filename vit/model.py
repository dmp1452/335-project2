import timm

def create_model(dataset_name, initializer=None):
    if dataset_name == "caltech":
        num_classes = 256
    elif dataset_name == "voc":
        num_classes = 20
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained = False,
        num_classes = num_classes
    )
    if initializer is not None:
        initializer(model)
    return model
