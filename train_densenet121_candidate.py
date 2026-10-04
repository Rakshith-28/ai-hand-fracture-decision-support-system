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
OUTPUT = PROJECT / "model_output_densenet121_candidate"
SEED = 211
BATCH_SIZE = 24
MAX_EPOCHS = 12
PATIENCE = 4
ORIGINAL_SOURCE_SHARE = 0.90

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
OUTPUT.mkdir(exist_ok=True)

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(7),
    transforms.RandomAffine(0, translate=(0.025, 0.025), scale=(0.97, 1.03)),
    transforms.RandomAutocontrast(p=0.2),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
fractured = train_data.class_to_idx["fractured"]
normal = train_data.class_to_idx["non_fractured"]

group_counts = {
    (class_index, added): sum(
        target == class_index and Path(path).name.startswith("bonefract_") == added
        for path, target in train_data.samples
    )
    for class_index in (fractured, normal)
    for added in (False, True)
}
weights = []
for path, target in train_data.samples:
    added = Path(path).name.startswith("bonefract_")
    source_share = 1.0 - ORIGINAL_SOURCE_SHARE if added else ORIGINAL_SOURCE_SHARE
    weights.append(0.5 * source_share / group_counts[(target, added)])
sampler = WeightedRandomSampler(
    weights, len(weights), replacement=True,
    generator=torch.Generator().manual_seed(SEED),
)
loaders = {
    "train": DataLoader(train_data, BATCH_SIZE, sampler=sampler, num_workers=0),
    "validation": DataLoader(validation_data, BATCH_SIZE, shuffle=False, num_workers=0),
    "test": DataLoader(test_data, BATCH_SIZE, shuffle=False, num_workers=0),
}

model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
for parameter in model.parameters():
    parameter.requires_grad = False
for parameter in model.features.denseblock4.parameters():
    parameter.requires_grad = True
for parameter in model.features.norm5.parameters():
    parameter.requires_grad = True
model.classifier = nn.Linear(model.classifier.in_features, 2)
criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=0.00012, weight_decay=0.0003
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.4, patience=1, min_lr=0.000005
)


def evaluate(loader):
    model.eval()
    matrix = [[0, 0], [0, 0]]
    loss_total = 0.0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images)
            loss_total += criterion(outputs, labels).item() * labels.size(0)
            predictions = outputs.argmax(1)
            total += labels.size(0)
            for actual, predicted in zip(labels.tolist(), predictions.tolist()):
                matrix[actual][predicted] += 1
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
best_score = (-1.0, -1.0)
best_epoch = 0
stale = 0
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    loss_total = 0.0
    correct = total = 0
    for images, labels in loaders["train"]:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_total += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    validation = evaluate(loaders["validation"])
    scheduler.step(validation["accuracy"])
    row = {
        "epoch": epoch,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "train_loss": loss_total / total,
        "train_accuracy": correct / total,
        **{f"validation_{key}": value for key, value in validation.items() if key != "matrix"},
    }
    history.append(row)
    print(json.dumps(row), flush=True)
    score = (validation["accuracy"], validation["fracture_f1"])
    if score > best_score:
        best_score = score
        best_epoch = epoch
        stale = 0
        torch.save({
            "model_state": model.state_dict(),
            "class_to_idx": train_data.class_to_idx,
            "architecture": "densenet121",
            "image_size": 224,
            "best_epoch": epoch,
        }, OUTPUT / "best_model.pt")
    else:
        stale += 1
        if stale >= PATIENCE:
            print(f"EARLY_STOP epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

checkpoint = torch.load(OUTPUT / "best_model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model_state"])
test = evaluate(loaders["test"])
with (OUTPUT / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)
(OUTPUT / "test_metrics.json").write_text(json.dumps({
    "model": "DenseNet-121 candidate",
    "seed": SEED,
    "best_epoch": best_epoch,
    "validation_selection": {"accuracy": best_score[0], "fracture_f1": best_score[1]},
    "test": test,
}, indent=2), encoding="utf-8")
print("TEST", json.dumps(test), flush=True)
