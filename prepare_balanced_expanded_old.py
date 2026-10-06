import csv
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance


PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "hand_dataset_prepared"
OUTPUT = PROJECT / "hand_dataset_balanced_expanded_old"
STAGING = PROJECT / "hand_dataset_balanced_expanded_old_staging"
MANIFEST = PROJECT / "dataset_metadata" / "balanced_expanded_old_manifest.csv"
SEED = 42


def old_files(split, class_name):
    return sorted(
        path for path in (SOURCE / split / class_name).iterdir()
        if path.is_file() and not path.name.startswith("bonefract_")
    )


def augment(source, destination, rng):
    with Image.open(source) as image:
        image = image.convert("L")
        angle = rng.uniform(-6.0, 6.0)
        shift_x = rng.randint(-5, 5)
        shift_y = rng.randint(-5, 5)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        image = image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1, 0, shift_x, 0, 1, shift_y),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.90, 1.10))
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.92, 1.08))
        image.save(destination, format="PNG", optimize=True)


def main():
    if OUTPUT.exists():
        raise RuntimeError(f"Expanded dataset already exists: {OUTPUT}")
    if STAGING.exists():
        raise RuntimeError(f"Staging folder already exists: {STAGING}")

    rng = random.Random(SEED)
    records = []
    try:
        for split in ("validation", "test"):
            for class_name in ("fractured", "non_fractured"):
                files = old_files(split, class_name)
                target = min(len(old_files(split, "fractured")), len(old_files(split, "non_fractured")))
                selected = rng.sample(files, target)
                destination = STAGING / split / class_name
                destination.mkdir(parents=True, exist_ok=True)
                for source in selected:
                    shutil.copy2(source, destination / source.name)
                    records.append((split, class_name, "original", source.name, source.name))

        fractured = old_files("train", "fractured")
        healthy = old_files("train", "non_fractured")
        target = len(healthy)
        for class_name, files in (("fractured", fractured), ("non_fractured", healthy)):
            destination = STAGING / "train" / class_name
            destination.mkdir(parents=True, exist_ok=True)
            for source in files:
                shutil.copy2(source, destination / source.name)
                records.append(("train", class_name, "original", source.name, source.name))

        needed = target - len(fractured)
        for number in range(needed):
            source = fractured[number % len(fractured)]
            filename = f"aug_{number + 1:04d}_{source.stem}.png"
            augment(source, STAGING / "train" / "fractured" / filename, rng)
            records.append(("train", "fractured", "augmented", filename, source.name))

        STAGING.replace(OUTPUT)
        MANIFEST.parent.mkdir(exist_ok=True)
        with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["split", "class", "kind", "filename", "source_filename"])
            writer.writerows(records)
        print(f"Training: {target} fractured ({len(fractured)} original + {needed} augmented) + {target} non-fractured")
        print("Validation: 77 fractured + 77 non-fractured")
        print("Test: 77 fractured + 77 non-fractured")
        print(f"Dataset: {OUTPUT}")
    except Exception:
        if STAGING.exists():
            shutil.rmtree(STAGING)
        raise


if __name__ == "__main__":
    main()
