import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "hand_dataset_prepared"
OUTPUT = PROJECT / "model_output"
SEED = 42
BATCH_SIZE = 32
EPOCHS = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
OUTPUT.mkdir(parents=True, exist_ok=True)

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(7),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

datasets_by_split = {
    "train": datasets.ImageFolder(DATA / "train", transform=train_transform),
    "validation": datasets.ImageFolder(DATA / "validation", transform=eval_transform),
    "test": datasets.ImageFolder(DATA / "test", transform=eval_transform),
}
loaders = {
    name: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=name == "train", num_workers=0)
    for name, ds in datasets_by_split.items()
}

device = torch.device("cpu")
weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)
for parameter in model.features.parameters():
    parameter.requires_grad = False
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model.to(device)

train_targets = datasets_by_split["train"].targets
counts = np.bincount(train_targets, minlength=2)
class_weights = torch.tensor(len(train_targets) / (2 * counts), dtype=torch.float32, device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)


def evaluate(loader):
    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0
    matrix = [[0, 0], [0, 0]]
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_total += criterion(outputs, labels).item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            for actual, predicted in zip(labels.tolist(), predictions.tolist()):
                matrix[actual][predicted] += 1
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "loss": loss_total / total,
        "accuracy": correct / total,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "confusion_matrix": matrix,
    }


history = []
best_validation_loss = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loaders["train"]:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    validation = evaluate(loaders["validation"])
    row = {
        "epoch": epoch,
        "train_loss": running_loss / total,
        "train_accuracy": correct / total,
        "validation_loss": validation["loss"],
        "validation_accuracy": validation["accuracy"],
        "validation_f1": validation["f1"],
    }
    history.append(row)
    print(json.dumps(row), flush=True)
    if validation["loss"] < best_validation_loss:
        best_validation_loss = validation["loss"]
        torch.save({
            "model_state": model.state_dict(),
            "class_to_idx": datasets_by_split["train"].class_to_idx,
            "architecture": "mobilenet_v3_small",
            "image_size": 224,
        }, OUTPUT / "best_model.pt")

checkpoint = torch.load(OUTPUT / "best_model.pt", map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state"])
test_metrics = evaluate(loaders["test"])

with (OUTPUT / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)

summary = {
    "model": "MobileNetV3-Small transfer learning baseline",
    "seed": SEED,
    "epochs": EPOCHS,
    "class_to_idx": datasets_by_split["train"].class_to_idx,
    "class_weights": class_weights.tolist(),
    "test": test_metrics,
}
with (OUTPUT / "test_metrics.json").open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2)

print("TEST", json.dumps(test_metrics), flush=True)
