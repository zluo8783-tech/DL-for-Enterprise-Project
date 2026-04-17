import torch
import torch.nn as nn
from torchvision.models import vit_b_16
import timm


def build_model(model_name, num_classes):
    if model_name == "resnet":
        model = timm.create_model("resnet50", pretrained=False)
    elif model_name == "resmlp":
        model = timm.create_model("resmlp_12_224", pretrained=False)
    elif model_name == "vit":
        model = vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(model_name)

    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "head"):
        model.head = nn.Linear(model.head.in_features, num_classes)

    return model


def load_model(path, device, model_name):
    checkpoint = torch.load(path, map_location=device)

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = build_model(model_name, num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, model_name