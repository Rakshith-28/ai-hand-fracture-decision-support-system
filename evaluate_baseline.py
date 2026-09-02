import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "hand_dataset_prepared" / "test"
OUTPUT = PROJECT / "model_output"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
dataset = datasets.ImageFolder(DATA, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

checkpoint = torch.load(OUTPUT / "best_model.pt", map_location="cpu", weights_only=True)
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model.load_state_dict(checkpoint["model_state"])
model.eval()

matrix = [[0, 0], [0, 0]]
with torch.no_grad():
    for images, labels in loader:
        predictions = model(images).argmax(dim=1)
        for actual, predicted in zip(labels.tolist(), predictions.tolist()):
            matrix[actual][predicted] += 1

fractured_index = dataset.class_to_idx["fractured"]
normal_index = dataset.class_to_idx["non_fractured"]
tp = matrix[fractured_index][fractured_index]
fn = matrix[fractured_index][normal_index]
fp = matrix[normal_index][fractured_index]
tn = matrix[normal_index][normal_index]
accuracy = (tp + tn) / sum(sum(row) for row in matrix)
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
specificity = tn / (tn + fp) if tn + fp else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

metrics = {
    "positive_class": "fractured",
    "class_to_idx": dataset.class_to_idx,
    "accuracy": accuracy,
    "fracture_precision": precision,
    "fracture_recall_sensitivity": recall,
    "non_fracture_specificity": specificity,
    "fracture_f1": f1,
    "confusion_matrix_rows_actual_columns_predicted": matrix,
    "true_fractured": tp,
    "missed_fractured": fn,
    "false_fracture_alarms": fp,
    "true_non_fractured": tn,
}
with (OUTPUT / "corrected_test_metrics.json").open("w", encoding="utf-8") as stream:
    json.dump(metrics, stream, indent=2)
print(json.dumps(metrics, indent=2))
