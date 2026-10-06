import csv
import hashlib
import io
import random
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageOps


PROJECT = Path(__file__).resolve().parent
DATASET = PROJECT / "hand_dataset"
ARCHIVE = PROJECT / "raw_data" / "BoneFract.zip"
EXCLUDED = PROJECT / "excluded_evaluation_images"
TRAIN_MANIFEST = PROJECT / "additional_fractured_manifest.csv"
OUTPUT_MANIFEST = PROJECT / "evaluation_balance_manifest.csv"
TARGET_PER_CLASS = 177
SEED = 2026


def standardized_png(raw_bytes):
    with Image.open(io.BytesIO(raw_bytes)) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
        return output.getvalue()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_nonfractured():
    for split in ("validation", "test"):
        source = EXCLUDED / split / "non_fractured"
        destination = DATASET / split / "non_fractured"
        destination.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            for path in source.iterdir():
                if path.is_file():
                    shutil.move(str(path), destination / path.name)


def remove_cross_split_duplicates():
    seen = {}
    removed = []
    # Training takes precedence; validation takes precedence over test.
    for split in ("train", "validation", "test"):
        for label in ("fractured", "non_fractured"):
            folder = DATASET / split / label
            for path in sorted(folder.iterdir()):
                if not path.is_file():
                    continue
                file_digest = digest(path)
                if file_digest not in seen:
                    seen[file_digest] = (split, label, path)
                    continue
                destination = EXCLUDED / "duplicates" / split / label / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), destination)
                removed.append((split, label, path.name, file_digest))
    return set(seen), removed


def used_source_members():
    if not TRAIN_MANIFEST.is_file():
        return set()
    with TRAIN_MANIFEST.open(newline="", encoding="utf-8") as stream:
        return {row["source_member"] for row in csv.DictReader(stream)}


def candidate_label(member):
    parts = member.split("/")
    if len(parts) != 6 or parts[2] not in {"Hand", "wrist"}:
        return None
    if not member.lower().endswith(".png"):
        return None
    if parts[4] == "Positive":
        return "fractured"
    if parts[4] == "Negative":
        return "non_fractured"
    return None


def fill_splits(existing_hashes):
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"Missing archive: {ARCHIVE}")
    already_used = used_source_members()
    rng = random.Random(SEED)
    manifest_rows = []

    with zipfile.ZipFile(ARCHIVE) as archive:
        candidates = {"fractured": [], "non_fractured": []}
        for member in archive.namelist():
            label = candidate_label(member)
            if label and member not in already_used:
                candidates[label].append(member)
        for members in candidates.values():
            rng.shuffle(members)

        candidate_positions = {"fractured": 0, "non_fractured": 0}
        for split, target in (("train", 826), ("validation", 177), ("test", 177)):
            for label in ("fractured", "non_fractured"):
                destination = DATASET / split / label
                destination.mkdir(parents=True, exist_ok=True)
                needed = target - sum(path.is_file() for path in destination.iterdir())
                while needed > 0:
                    position = candidate_positions[label]
                    if position >= len(candidates[label]):
                        raise RuntimeError(f"Not enough unique {label} images in the archive")
                    member = candidates[label][position]
                    candidate_positions[label] += 1
                    data = standardized_png(archive.read(member))
                    file_digest = hashlib.sha256(data).hexdigest()
                    filename = Path(member).name
                    output_path = destination / filename
                    if file_digest in existing_hashes or output_path.exists():
                        continue
                    output_path.write_bytes(data)
                    existing_hashes.add(file_digest)
                    parts = member.split("/")
                    manifest_rows.append({
                        "split": split,
                        "label": label,
                        "filename": filename,
                        "anatomy": parts[2],
                        "patient_id": parts[3],
                        "source_member": member,
                        "processed_sha256": file_digest,
                    })
                    needed -= 1
    return manifest_rows


def verify():
    hashes_by_split = {}
    for split, expected in (("train", 826), ("validation", 177), ("test", 177)):
        for label in ("fractured", "non_fractured"):
            files = [path for path in (DATASET / split / label).iterdir() if path.is_file()]
            if len(files) != expected:
                raise RuntimeError(f"Wrong count for {split}/{label}: {len(files)}")
            for path in files:
                file_digest = digest(path)
                previous = hashes_by_split.get(file_digest)
                if previous and previous != split:
                    raise RuntimeError(f"Cross-split duplicate remains: {path}")
                hashes_by_split[file_digest] = split


def main():
    restore_nonfractured()
    hashes, removed = remove_cross_split_duplicates()
    rows = fill_splits(hashes)
    verify()
    with OUTPUT_MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "split", "label", "filename", "anatomy", "patient_id",
            "source_member", "processed_sha256",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Removed {len(removed)} cross-split duplicate files")
    print(f"Added {len(rows)} unique BoneFract files")
    print(f"Manifest: {OUTPUT_MANIFEST}")
    print("Final counts: train 826+826, validation 177+177, test 177+177")


if __name__ == "__main__":
    main()
