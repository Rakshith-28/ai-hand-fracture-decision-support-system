import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models

from tune_threshold import DATA, BATCH_SIZE, calculate, collect, image_transform, load_model


PROJECT = Path(__file__).resolve().parent
RESNET_PATH = PROJECT / "model_output_resnet18_candidate" / "best_model.pt"
OUTPUT = PROJECT / "model_output_advanced" / "ensemble_evaluation.json"


def load_resnet():
    checkpoint = torch.load(RESNET_PATH, map_location="cpu", weights_only=True)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main():
    mobile, class_to_idx = load_model()
    resnet = load_resnet()
    fractured = class_to_idx["fractured"]
    normal = class_to_idx["non_fractured"]
    validation_loader = DataLoader(
        datasets.ImageFolder(DATA / "validation", transform=image_transform(224)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    mobile_val, labels = collect(mobile, validation_loader)
    resnet_val, _ = collect(resnet, validation_loader)
    candidates = []
    for mobile_weight in np.arange(0.5, 1.001, 0.05):
        combined = mobile_weight * mobile_val + (1.0 - mobile_weight) * resnet_val
        for threshold in np.arange(0.30, 0.651, 0.01):
            metrics = calculate(combined, labels, fractured, normal, float(threshold))
            candidates.append((
                metrics["accuracy"], metrics["fracture_f1"],
                -abs(threshold - 0.5), mobile_weight, threshold, metrics,
            ))
    _, _, _, mobile_weight, threshold, validation = max(candidates)

    test_loader = DataLoader(
        datasets.ImageFolder(DATA / "test", transform=image_transform(224)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    mobile_test, test_labels = collect(mobile, test_loader)
    resnet_test, _ = collect(resnet, test_loader)
    combined_test = mobile_weight * mobile_test + (1.0 - mobile_weight) * resnet_test
    test = calculate(combined_test, test_labels, fractured, normal, float(threshold))
    report = {
        "selection_rule": "maximum validation accuracy; validation F1 breaks ties",
        "mobilenet_weight": float(mobile_weight),
        "resnet18_weight": float(1.0 - mobile_weight),
        "fracture_threshold": float(threshold),
        "validation": validation,
        "test": test,
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
