import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "bonefract_single_source"
OUTPUT = PROJECT / "model_output_single_source"
SEED = 2026
BATCH_SIZE = 24
MAX_EPOCHS = 15
PATIENCE = 4
ANATOMY_LOSS_WEIGHT = 0.25

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
OUTPUT.mkdir(exist_ok=True)

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(7),
    transforms.RandomAffine(0, translate=(0.03, 0.03), scale=(0.96, 1.04)),
    transforms.RandomAutocontrast(p=0.2),
    transforms.ColorJitter(brightness=0.10, contrast=0.10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class AnatomyDataset(Dataset):
    def __init__(self, folder, transform):
        self.base = datasets.ImageFolder(folder, transform=transform)
        self.class_to_idx = self.base.class_to_idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, fracture = self.base[index]
        filename = Path(self.base.samples[index][0]).name
        anatomy = 0 if filename.startswith("bonefract_hand_") else 1
        return image, fracture, anatomy


class MultiTaskMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.features = base.features
        self.avgpool = base.avgpool
        self.fracture_head = base.classifier
        self.fracture_head[3] = nn.Linear(self.fracture_head[3].in_features, 2)
        self.anatomy_head = nn.Sequential(nn.Linear(960, 256), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(256, 2))

    def forward(self, images):
        features = self.features(images)
        features = torch.flatten(self.avgpool(features), 1)
        return self.fracture_head(features), self.anatomy_head(features)


train_data = AnatomyDataset(DATA / "train", train_transform)
validation_data = AnatomyDataset(DATA / "validation", eval_transform)
test_data = AnatomyDataset(DATA / "test", eval_transform)
loaders = {
    "train": DataLoader(train_data, BATCH_SIZE, shuffle=True, num_workers=0),
    "validation": DataLoader(validation_data, BATCH_SIZE, shuffle=False, num_workers=0),
    "test": DataLoader(test_data, BATCH_SIZE, shuffle=False, num_workers=0),
}
model = MultiTaskMobileNet()
for parameter in model.features.parameters():
    parameter.requires_grad = False
for block in model.features[-6:]:
    for parameter in block.parameters():
        parameter.requires_grad = True
criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.00012, weight_decay=0.00025)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.4, patience=1, min_lr=0.000005)


def evaluate(loader):
    model.eval()
    matrix = [[0, 0], [0, 0]]
    anatomy_correct = total = 0
    with torch.inference_mode():
        for images, fractures, anatomies in loader:
            fracture_logits, anatomy_logits = model(images)
            predictions = fracture_logits.argmax(1)
            anatomy_correct += (anatomy_logits.argmax(1) == anatomies).sum().item()
            total += fractures.size(0)
            for actual, predicted in zip(fractures.tolist(), predictions.tolist()):
                matrix[actual][predicted] += 1
    fractured = loader.dataset.class_to_idx["fractured"]
    normal = loader.dataset.class_to_idx["non_fractured"]
    tp, fn = matrix[fractured][fractured], matrix[fractured][normal]
    fp, tn = matrix[normal][fractured], matrix[normal][normal]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / total, "fracture_precision": precision, "fracture_recall": recall,
            "specificity": specificity, "fracture_f1": f1, "anatomy_accuracy": anatomy_correct / total, "matrix": matrix}


history, best_score, stale = [], (-1.0, -1.0), 0
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    loss_total = correct = anatomy_correct = total = 0
    for images, fractures, anatomies in loaders["train"]:
        optimizer.zero_grad()
        fracture_logits, anatomy_logits = model(images)
        loss = criterion(fracture_logits, fractures) + ANATOMY_LOSS_WEIGHT * criterion(anatomy_logits, anatomies)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_total += loss.item() * fractures.size(0)
        correct += (fracture_logits.argmax(1) == fractures).sum().item()
        anatomy_correct += (anatomy_logits.argmax(1) == anatomies).sum().item()
        total += fractures.size(0)
    validation = evaluate(loaders["validation"])
    scheduler.step(validation["accuracy"])
    row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train_loss": loss_total / total,
           "train_accuracy": correct / total, "train_anatomy_accuracy": anatomy_correct / total,
           **{f"validation_{k}": v for k, v in validation.items() if k != "matrix"}}
    history.append(row)
    print(json.dumps(row), flush=True)
    score = (validation["accuracy"], validation["fracture_f1"])
    if score > best_score:
        best_score, stale = score, 0
        torch.save({"model_state": model.state_dict(), "class_to_idx": train_data.class_to_idx,
                    "architecture": "multitask_mobilenet_v3_large", "best_epoch": epoch}, OUTPUT / "best_model.pt")
    else:
        stale += 1
        if stale >= PATIENCE:
            break

checkpoint = torch.load(OUTPUT / "best_model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model_state"])
test = evaluate(loaders["test"])
with (OUTPUT / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
(OUTPUT / "test_metrics.json").write_text(json.dumps({"model": "Single-source anatomy-aware MobileNetV3-Large",
    "seed": SEED, "best_epoch": checkpoint["best_epoch"], "test": test}, indent=2), encoding="utf-8")
print("TEST", json.dumps(test), flush=True)
