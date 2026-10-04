import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "hand_dataset_prepared"
MODEL_PATH = PROJECT / "model_output_advanced" / "best_advanced_model.pt"
OUTPUT_PATH = PROJECT / "model_output_advanced" / "decision_threshold.json"
BATCH_SIZE = 24
MINIMUM_VALIDATION_RECALL = 0.72

def image_transform(size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["class_to_idx"]


def collect(model, loader):
    probabilities = []
    labels = []
    with torch.inference_mode():
        for images, batch_labels in loader:
            probabilities.append(torch.softmax(model(images), dim=1))
            labels.append(batch_labels)
    return torch.cat(probabilities), torch.cat(labels)


def calculate(probabilities, labels, fractured, normal, threshold):
    predictions = torch.where(
        probabilities[:, fractured] >= threshold,
        torch.tensor(fractured),
        torch.tensor(normal),
    )
    tp = int(((labels == fractured) & (predictions == fractured)).sum())
    fn = int(((labels == fractured) & (predictions == normal)).sum())
    fp = int(((labels == normal) & (predictions == fractured)).sum())
    tn = int(((labels == normal) & (predictions == normal)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(labels),
        "fracture_precision": precision,
        "fracture_recall": recall,
        "specificity": specificity,
        "fracture_f1": f1,
        "matrix": [[tp, fn], [fp, tn]],
    }


def main():
    model, class_to_idx = load_model()
    fractured = class_to_idx["fractured"]
    normal = class_to_idx["non_fractured"]
    candidates = []
    for image_size in (224, 288, 320):
        validation_loader = DataLoader(
            datasets.ImageFolder(DATA / "validation", transform=image_transform(image_size)),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        validation_probabilities, validation_labels = collect(model, validation_loader)
        for threshold in np.arange(0.20, 0.701, 0.005):
            result = calculate(
                validation_probabilities,
                validation_labels,
                fractured,
                normal,
                float(threshold),
            )
            if result["fracture_recall"] >= MINIMUM_VALIDATION_RECALL:
                candidates.append((result["accuracy"], result["fracture_f1"], image_size, float(threshold), result))
    if not candidates:
        raise RuntimeError("No threshold met the validation recall requirement")

    _, _, selected_size, threshold, validation_result = max(candidates)
    test_loader = DataLoader(
        datasets.ImageFolder(DATA / "test", transform=image_transform(selected_size)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_probabilities, test_labels = collect(model, test_loader)
    test_result = calculate(
        test_probabilities, test_labels, fractured, normal, threshold
    )
    report = {
        "selection_rule": "maximum validation accuracy with fracture recall >= 0.72",
        "image_size": selected_size,
        "fracture_threshold": threshold,
        "validation": validation_result,
        "test": test_result,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
