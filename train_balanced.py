import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms


PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "hand_dataset_prepared"
OUTPUT = PROJECT / "model_output_balanced"
SEED = 42
BATCH_SIZE = 32
EPOCHS = 8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
OUTPUT.mkdir(parents=True, exist_ok=True)

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_data = datasets.ImageFolder(DATA / "train", transform=train_transform)
validation_data = datasets.ImageFolder(DATA / "validation", transform=eval_transform)
test_data = datasets.ImageFolder(DATA / "test", transform=eval_transform)
counts = np.bincount(train_data.targets, minlength=2)
sample_weights = [1.0 / counts[target] for target in train_data.targets]
generator = torch.Generator().manual_seed(SEED)
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)
loaders = {
    "train": DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0),
    "validation": DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    "test": DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
}

model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
for parameter in model.features.parameters():
    parameter.requires_grad = False
for block in model.features[-3:]:
    for parameter in block.parameters():
        parameter.requires_grad = True
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    [parameter for parameter in model.parameters() if parameter.requires_grad],
    lr=0.0003,
    weight_decay=0.0001,
)


def evaluate(loader):
    model.eval()
    matrix = [[0, 0], [0, 0]]
    loss_total = 0.0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss_total += criterion(outputs, labels).item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            for actual, predicted in zip(labels.tolist(), predictions.tolist()):
                matrix[actual][predicted] += 1

    fractured = loader.dataset.class_to_idx["fractured"]
    normal = loader.dataset.class_to_idx["non_fractured"]
    tp = matrix[fractured][fractured]
    fn = matrix[fractured][normal]
    fp = matrix[normal][fractured]
    tn = matrix[normal][normal]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "loss": loss_total / total,
        "accuracy": (tp + tn) / total,
        "fracture_precision": precision,
        "fracture_recall": recall,
        "specificity": specificity,
        "fracture_f1": f1,
        "matrix": matrix,
    }


history = []
best_f1 = -1.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loaders["train"]:
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
        "validation_fracture_recall": validation["fracture_recall"],
        "validation_fracture_f1": validation["fracture_f1"],
    }
    history.append(row)
    print(json.dumps(row), flush=True)
    if validation["fracture_f1"] > best_f1:
        best_f1 = validation["fracture_f1"]
        torch.save({
            "model_state": model.state_dict(),
            "class_to_idx": train_data.class_to_idx,
            "architecture": "mobilenet_v3_small",
            "image_size": 224,
            "balanced_sampling": True,
        }, OUTPUT / "best_balanced_model.pt")

checkpoint = torch.load(OUTPUT / "best_balanced_model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model_state"])
test = evaluate(loaders["test"])

with (OUTPUT / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)
with (OUTPUT / "test_metrics.json").open("w", encoding="utf-8") as stream:
    json.dump({"model": "Balanced MobileNetV3-Small", "seed": SEED, "epochs": EPOCHS, "test": test}, stream, indent=2)
print("TEST", json.dumps(test), flush=True)
