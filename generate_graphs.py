import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parent
ADVANCED = PROJECT / "model_output_advanced"
BALANCED = PROJECT / "reference_results"
OUTPUT = PROJECT / "graph_outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)


def read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


with (ADVANCED / "training_history.csv").open(
    "r", newline="", encoding="utf-8-sig"
) as stream:
    history = list(csv.DictReader(stream))

epochs = [int(row["epoch"]) for row in history]
train_accuracy = [float(row["train_accuracy"]) * 100 for row in history]
validation_accuracy = [float(row["validation_accuracy"]) * 100 for row in history]
train_loss = [float(row["train_loss"]) for row in history]
validation_loss = [float(row["validation_loss"]) for row in history]

advanced = read_json(ADVANCED / "test_metrics.json")
balanced = read_json(BALANCED / "test_metrics.json")

plt.style.use("seaborn-v0_8-whitegrid")


def finish(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT / filename, dpi=300, bbox_inches="tight")
    plt.close()


# 1. Accuracy graph
plt.figure(figsize=(9, 5.5))
plt.plot(epochs, train_accuracy, marker="o", linewidth=2, label="Training accuracy")
plt.plot(epochs, validation_accuracy, marker="o", linewidth=2, label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.xticks(epochs)
plt.ylim(0, 100)
plt.legend()
finish("accuracy_graph.png")


# 2. Error/loss graph
plt.figure(figsize=(9, 5.5))
plt.plot(epochs, train_loss, marker="o", linewidth=2, label="Training loss")
plt.plot(epochs, validation_loss, marker="o", linewidth=2, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Training and Validation Error (Loss)")
plt.xticks(epochs)
plt.legend()
finish("error_loss_graph.png")


# 3. Confusion matrix
matrix = np.array(advanced["test"]["matrix"])
fig, axis = plt.subplots(figsize=(7, 6))
image = axis.imshow(matrix, cmap="Blues")
axis.set_title("Advanced Model Confusion Matrix")
axis.set_xlabel("Predicted class")
axis.set_ylabel("Actual class")
axis.set_xticks([0, 1], ["Fractured", "Non-fractured"])
axis.set_yticks([0, 1], ["Fractured", "Non-fractured"])
for row in range(2):
    for column in range(2):
        color = "white" if matrix[row, column] > matrix.max() / 2 else "black"
        axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                  fontsize=18, fontweight="bold", color=color)
fig.colorbar(image, ax=axis)
finish("confusion_matrix.png")


# 4. Model comparison graph
metric_names = ["Accuracy", "Precision", "Recall", "Specificity", "F1-score"]
metric_keys = [
    "accuracy",
    "fracture_precision",
    "fracture_recall",
    "specificity",
    "fracture_f1",
]
balanced_values = [balanced["test"][key] * 100 for key in metric_keys]
advanced_values = [advanced["test"][key] * 100 for key in metric_keys]

x = np.arange(len(metric_names))
width = 0.36
fig, axis = plt.subplots(figsize=(10, 5.8))
old_bars = axis.bar(x - width / 2, balanced_values, width,
                    label="MobileNetV3-Small", color="#7CA6D8")
new_bars = axis.bar(x + width / 2, advanced_values, width,
                    label="MobileNetV3-Large", color="#1F5FA7")
axis.set_ylabel("Score (%)")
axis.set_title("Recorded Test Metric Comparison")
axis.set_xticks(x, metric_names)
axis.set_ylim(0, 100)
axis.legend()
axis.bar_label(old_bars, fmt="%.1f", padding=3, fontsize=9)
axis.bar_label(new_bars, fmt="%.1f", padding=3, fontsize=9)
finish("model_comparison.png")

print(f"Graphs saved in: {OUTPUT}")
for file in sorted(OUTPUT.glob("*.png")):
    print(file.name)
