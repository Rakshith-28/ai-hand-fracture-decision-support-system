import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from tune_threshold import DATA, BATCH_SIZE, calculate, collect, image_transform, load_model


OUTPUT = Path(__file__).resolve().parent / "model_output_advanced" / "accuracy_threshold.json"


def main():
    model, class_to_idx = load_model()
    fractured = class_to_idx["fractured"]
    normal = class_to_idx["non_fractured"]
    validation_loader = DataLoader(
        datasets.ImageFolder(DATA / "validation", transform=image_transform(224)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    probabilities, labels = collect(model, validation_loader)
    candidates = []
    for threshold in np.arange(0.20, 0.801, 0.0025):
        result = calculate(probabilities, labels, fractured, normal, float(threshold))
        candidates.append((result["accuracy"], result["fracture_f1"], -abs(threshold - 0.5), threshold, result))
    _, _, _, threshold, validation = max(candidates)

    test_loader = DataLoader(
        datasets.ImageFolder(DATA / "test", transform=image_transform(224)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_probabilities, test_labels = collect(model, test_loader)
    test = calculate(test_probabilities, test_labels, fractured, normal, float(threshold))
    report = {
        "selection_rule": "maximum validation accuracy; F1 and closeness to 0.5 break ties",
        "fracture_threshold": float(threshold),
        "validation": validation,
        "test": test,
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
