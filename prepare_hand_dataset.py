import csv
import random
from pathlib import Path

from PIL import Image, ImageFile, ImageOps


PROJECT = Path(__file__).resolve().parent
SOURCE_CSV = PROJECT / "hand_dataset.csv"
SOURCE_IMAGES = PROJECT / "hand_dataset_images"
OUTPUT = PROJECT / "hand_dataset_prepared"
SEED = 42
ImageFile.LOAD_TRUNCATED_IMAGES = True


def split_rows(rows):
    rng = random.Random(SEED)
    groups = {"0": [], "1": []}
    for row in rows:
        groups[row["fractured"]].append(row)

    splits = {"train": [], "validation": [], "test": []}
    for group in groups.values():
        rng.shuffle(group)
        total = len(group)
        train_end = round(total * 0.70)
        validation_end = train_end + round(total * 0.15)
        splits["train"].extend(group[:train_end])
        splits["validation"].extend(group[train_end:validation_end])
        splits["test"].extend(group[validation_end:])

    for split_rows_list in splits.values():
        rng.shuffle(split_rows_list)
    return splits


def prepare_image(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        image = ImageOps.pad(image, (224, 224), color=0, method=Image.Resampling.LANCZOS)
        image.save(destination, format="PNG", optimize=True)


with SOURCE_CSV.open("r", newline="", encoding="utf-8-sig") as stream:
    rows = list(csv.DictReader(stream))

splits = split_rows(rows)
fieldnames = list(rows[0].keys()) + ["split", "processed_image"]

for split_name, split_data in splits.items():
    output_rows = []
    for row in split_data:
        class_name = "fractured" if row["fractured"] == "1" else "non_fractured"
        source = SOURCE_IMAGES / class_name / row["image_id"]
        relative_output = Path(split_name) / class_name / f"{Path(row['image_id']).stem}.png"
        prepare_image(source, OUTPUT / relative_output)
        output_rows.append({**row, "split": split_name, "processed_image": str(relative_output)})

    with (OUTPUT / f"{split_name}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

print("Prepared dataset:", OUTPUT)
for split_name in ("train", "validation", "test"):
    fractured = sum(row["fractured"] == "1" for row in splits[split_name])
    non_fractured = sum(row["fractured"] == "0" for row in splits[split_name])
    print(split_name, len(splits[split_name]), fractured, non_fractured)
