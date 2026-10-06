import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT / "graph_data"
OUTPUT = PROJECT / "graph_outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)


def read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


with (MODEL_DIR / "training_history.csv").open(
    "r", newline="", encoding="utf-8-sig"
) as stream:
    history = list(csv.DictReader(stream))

epochs = [int(row["epoch"]) for row in history]
train_accuracy = [float(row["train_accuracy"]) * 100 for row in history]
validation_accuracy = [float(row["validation_accuracy"]) * 100 for row in history]
train_loss = [float(row["train_loss"]) for row in history]
has_validation_loss = "validation_loss" in history[0]
validation_loss = [float(row["validation_loss"]) for row in history] if has_validation_loss else None

metrics = read_json(MODEL_DIR / "test_metrics.json")

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
if has_validation_loss:
    plt.plot(epochs, train_loss, marker="o", linewidth=2, label="Training loss")
    plt.plot(epochs, validation_loss, marker="o", linewidth=2, label="Validation loss")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and Validation Error (Loss)")
else:
    train_error = [100 - value for value in train_accuracy]
    validation_error = [100 - value for value in validation_accuracy]
    plt.plot(epochs, train_error, marker="o", linewidth=2, label="Training error")
    plt.plot(epochs, validation_error, marker="o", linewidth=2, label="Validation error")
    plt.ylabel("Classification error (%)")
    plt.title("Training and Validation Classification Error")
plt.xlabel("Epoch")
plt.xticks(epochs)
plt.legend()
finish("error_loss_graph.png")


# 3. Confusion matrix
matrix = np.array(metrics["test"]["matrix"])
fig, axis = plt.subplots(figsize=(7, 6))
image = axis.imshow(matrix, cmap="Blues")
axis.set_title("Latest Balanced Model Confusion Matrix")
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


# 4. Dataset distribution
dataset = metrics["dataset"]
splits = ["Training", "Validation", "Test"]
fractured_counts = [dataset[name.lower()]["fractured"] for name in splits]
normal_counts = [dataset[name.lower()]["non_fractured"] for name in splits]
positions = np.arange(len(splits))
width = 0.36
plt.figure(figsize=(9, 5.5))
plt.bar(positions - width / 2, fractured_counts, width, label="Fractured")
plt.bar(positions + width / 2, normal_counts, width, label="Non-fractured")
plt.xlabel("Dataset split")
plt.ylabel("Number of X-rays")
plt.title("Latest Balanced Dataset Distribution")
plt.xticks(positions, splits)
plt.legend()
for position, value in zip(positions - width / 2, fractured_counts):
    plt.text(position, value + 10, str(value), ha="center")
for position, value in zip(positions + width / 2, normal_counts):
    plt.text(position, value + 10, str(value), ha="center")
finish("dataset_distribution.png")


print(f"Graphs saved in: {OUTPUT}")
for file in sorted(OUTPUT.glob("*.png")):
    print(file.name)
