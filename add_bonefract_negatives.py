import csv
import hashlib
import io
import random
import shutil
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps, UnidentifiedImageError
from remotezip import RemoteZip


PROJECT = Path(__file__).resolve().parent
DATASET = PROJECT / "hand_dataset_prepared"
DESTINATION = DATASET / "train" / "non_fractured"
METADATA = PROJECT / "dataset_metadata"
POSITIVE_MANIFEST = METADATA / "bonefract_added_manifest.csv"
NEGATIVE_MANIFEST = METADATA / "bonefract_negative_manifest.csv"
STAGING = PROJECT / "bonefract_negative_staging"
ARCHIVE_URL = "https://data.mendeley.com/public-files/datasets/4cr7f2359x/files/76c032f0-b8be-49b8-a943-d30d77ac62e4/file_downloaded"
SEED = 43
IMAGE_SIZE = 224


def standardized_jpeg(raw_bytes):
    with Image.open(io.BytesIO(raw_bytes)) as image:
        image.load()
        image = ImageOps.exif_transpose(image).convert("L")
        image = ImageOps.autocontrast(image)
        image = ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95, optimize=True)
        return output.getvalue()


def existing_hashes():
    hashes = set()
    for path in DATASET.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            converted = standardized_jpeg(path.read_bytes())
        except (UnidentifiedImageError, OSError):
            continue
        hashes.add(hashlib.sha256(converted).hexdigest())
    return hashes


def positive_patient_ids_and_targets():
    patients = set()
    targets = {"Hand": 0, "wrist": 0}
    with POSITIVE_MANIFEST.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            patients.add(row["patient_id"])
            anatomy = "Hand" if row["anatomy"].lower() == "hand" else "wrist"
            targets[anatomy] += 1
    return patients, targets


def select_members(names, excluded_patients, targets):
    candidates = {"Hand": [], "wrist": []}
    for name in names:
        path = PurePosixPath(name)
        parts = path.parts
        if len(parts) < 6 or path.suffix.lower() != ".png":
            continue
        if parts[1] != "train" or parts[4] != "Negative" or parts[2] not in candidates:
            continue
        if parts[3] not in excluded_patients:
            candidates[parts[2]].append(name)

    rng = random.Random(SEED)
    selected = []
    for anatomy, count in targets.items():
        rng.shuffle(candidates[anatomy])
        if len(candidates[anatomy]) < count:
            raise RuntimeError(f"Not enough {anatomy} negative images")
        selected.extend(candidates[anatomy][:count])
    rng.shuffle(selected)
    return selected


def main():
    if not POSITIVE_MANIFEST.is_file():
        raise FileNotFoundError(POSITIVE_MANIFEST)
    if list(DESTINATION.glob("bonefract_*.jpg")):
        raise RuntimeError("BoneFract negatives already exist; nothing changed.")
    if STAGING.exists():
        raise RuntimeError(f"Staging folder already exists: {STAGING}")

    excluded_patients, targets = positive_patient_ids_and_targets()
    known_hashes = existing_hashes()
    new_hashes = set()
    records = []
    STAGING.mkdir()
    try:
        with RemoteZip(ARCHIVE_URL) as archive:
            selected = select_members(archive.namelist(), excluded_patients, targets)
            for number, member in enumerate(selected, start=1):
                converted = standardized_jpeg(archive.read(member))
                digest = hashlib.sha256(converted).hexdigest()
                if digest in known_hashes or digest in new_hashes:
                    continue
                new_hashes.add(digest)
                source = PurePosixPath(member)
                anatomy, patient = source.parts[2].lower(), source.parts[3]
                filename = f"bonefract_{anatomy}_{patient}.jpg"
                (STAGING / filename).write_bytes(converted)
                records.append((filename, anatomy, patient, member))
                if number % 25 == 0 or number == len(selected):
                    print(f"Downloaded {number}/{len(selected)} ({len(records)} unique)", flush=True)

        if len(records) < 670:
            raise RuntimeError(f"Only {len(records)} unique negatives prepared")
        for path in STAGING.iterdir():
            shutil.move(str(path), str(DESTINATION / path.name))
        with NEGATIVE_MANIFEST.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["filename", "anatomy", "patient_id", "source_member"])
            writer.writerows(records)
        print(f"Added {len(records)} matched BoneFract negative images")
        print(f"Manifest: {NEGATIVE_MANIFEST}")
    finally:
        if STAGING.exists():
            shutil.rmtree(STAGING)


if __name__ == "__main__":
    main()
