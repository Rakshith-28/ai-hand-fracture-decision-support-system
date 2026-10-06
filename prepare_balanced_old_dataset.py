import csv
import random
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "hand_dataset_prepared"
OUTPUT = PROJECT / "hand_dataset_balanced_old"
STAGING = PROJECT / "hand_dataset_balanced_old_staging"
MANIFEST = PROJECT / "dataset_metadata" / "balanced_old_manifest.csv"
SEED = 42
SPLITS = ("train", "validation", "test")
CLASSES = ("fractured", "non_fractured")


def old_source_files(split, class_name):
    return sorted(
        path for path in (SOURCE / split / class_name).iterdir()
        if path.is_file() and not path.name.startswith("bonefract_")
    )


def main():
    if OUTPUT.exists():
        raise RuntimeError(f"Balanced dataset already exists: {OUTPUT}")
    if STAGING.exists():
        raise RuntimeError(f"Staging folder already exists: {STAGING}")

    rng = random.Random(SEED)
    records = []
    STAGING.mkdir()
    try:
        for split in SPLITS:
            fractured = old_source_files(split, "fractured")
            healthy = old_source_files(split, "non_fractured")
            target = min(len(fractured), len(healthy))
            selected = {
                "fractured": rng.sample(fractured, target),
                "non_fractured": rng.sample(healthy, target),
            }
            for class_name in CLASSES:
                destination = STAGING / split / class_name
                destination.mkdir(parents=True)
                for source in selected[class_name]:
                    shutil.copy2(source, destination / source.name)
                    records.append((split, class_name, source.name, str(source.relative_to(PROJECT))))
            print(f"{split}: {target} fractured + {target} non-fractured")

        STAGING.replace(OUTPUT)
        MANIFEST.parent.mkdir(exist_ok=True)
        with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["split", "class", "filename", "source_path"])
            writer.writerows(records)
        print(f"Prepared {len(records)} balanced old-source images")
        print(f"Dataset: {OUTPUT}")
    except Exception:
        if STAGING.exists():
            shutil.rmtree(STAGING)
        raise


if __name__ == "__main__":
    main()
