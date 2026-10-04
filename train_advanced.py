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
OUTPUT = PROJECT / "model_output_advanced"
SEED = 42
BATCH_SIZE = 24
MAX_EPOCHS = 15
PATIENCE = 5
FRACTURE_SAMPLING_TARGET = 0.50
ORIGINAL_SOURCE_SHARE = 0.90
ADDED_SOURCE_PREFIX = "bonefract_"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
OUTPUT.mkdir(parents=True, exist_ok=True)

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
    transforms.RandomAutocontrast(p=0.25),
    transforms.ColorJitter(brightness=0.12, contrast=0.12),
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
fractured_index = train_data.class_to_idx["fractured"]
normal_index = train_data.class_to_idx["non_fractured"]
group_counts = {
    (class_index, is_added): sum(
        target == class_index
        and Path(path).name.startswith(ADDED_SOURCE_PREFIX) == is_added
        for path, target in train_data.samples
    )
    for class_index in (fractured_index, normal_index)
    for is_added in (False, True)
}
weights = []
for path, target in train_data.samples:
    class_share = (
        FRACTURE_SAMPLING_TARGET
        if target == fractured_index
        else 1.0 - FRACTURE_SAMPLING_TARGET
    )
    is_added = Path(path).name.startswith(ADDED_SOURCE_PREFIX)
    source_share = 1.0 - ORIGINAL_SOURCE_SHARE if is_added else ORIGINAL_SOURCE_SHARE
    weight = class_share * source_share / group_counts[(target, is_added)]
    weights.append(weight)
sampler = WeightedRandomSampler(
    weights, len(weights), replacement=True,
    generator=torch.Generator().manual_seed(SEED),
)
loaders = {
    "train": DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0),
    "validation": DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    "test": DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
}

checkpoint_path = OUTPUT / "best_advanced_model.pt"
initial_checkpoint_used = checkpoint_path.is_file()
model = models.mobilenet_v3_large(
    weights=None if initial_checkpoint_used else models.MobileNet_V3_Large_Weights.DEFAULT
)
for parameter in model.features.parameters():
    parameter.requires_grad = False
for block in model.features[-3:]:
    for parameter in block.parameters():
        parameter.requires_grad = True
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
if initial_checkpoint_used:
    initial_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(initial_checkpoint["model_state"])

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(
    [parameter for parameter in model.parameters() if parameter.requires_grad],
    lr=0.00001 if initial_checkpoint_used else 0.00015,
    weight_decay=0.0002,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2, min_lr=0.000005,
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
    tp, fn = matrix[fractured][fractured], matrix[fractured][normal]
    fp, tn = matrix[normal][fractured], matrix[normal][normal]
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
if initial_checkpoint_used:
    initial_validation = evaluate(loaders["validation"])
    best_f1 = initial_validation["fracture_f1"]
    print("INITIAL_VALIDATION", json.dumps(initial_validation), flush=True)
else:
    best_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loaders["train"]:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    validation = evaluate(loaders["validation"])
    scheduler.step(validation["fracture_f1"])
    row = {
        "epoch": epoch,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "train_loss": running_loss / total,
        "train_accuracy": correct / total,
        "validation_loss": validation["loss"],
        "validation_accuracy": validation["accuracy"],
        "validation_fracture_precision": validation["fracture_precision"],
        "validation_fracture_recall": validation["fracture_recall"],
        "validation_specificity": validation["specificity"],
        "validation_fracture_f1": validation["fracture_f1"],
    }
    history.append(row)
    print(json.dumps(row), flush=True)

    if validation["fracture_f1"] > best_f1 + 0.001:
        best_f1 = validation["fracture_f1"]
        best_epoch = epoch
        epochs_without_improvement = 0
        torch.save({
            "model_state": model.state_dict(),
            "class_to_idx": train_data.class_to_idx,
            "architecture": "mobilenet_v3_large",
            "image_size": 224,
            "balanced_sampling": True,
            "fracture_sampling_target": FRACTURE_SAMPLING_TARGET,
            "original_source_share": ORIGINAL_SOURCE_SHARE,
            "best_epoch": best_epoch,
        }, OUTPUT / "best_advanced_model.pt")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"EARLY_STOP epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

checkpoint = torch.load(OUTPUT / "best_advanced_model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model_state"])
test = evaluate(loaders["test"])

with (OUTPUT / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)
with (OUTPUT / "test_metrics.json").open("w", encoding="utf-8") as stream:
    json.dump({
        "model": "Advanced MobileNetV3-Large",
        "seed": SEED,
        "maximum_epochs": MAX_EPOCHS,
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "fracture_sampling_target": FRACTURE_SAMPLING_TARGET,
        "original_source_share": ORIGINAL_SOURCE_SHARE,
        "initial_checkpoint_used": initial_checkpoint_used,
        "test": test,
    }, stream, indent=2)
print("TEST", json.dumps(test), flush=True)
