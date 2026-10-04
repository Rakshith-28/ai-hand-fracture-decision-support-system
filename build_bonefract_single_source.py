import csv
import hashlib
import io
import random
import time
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps
from remotezip import RemoteZip


PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "bonefract_single_source"
MANIFEST = PROJECT / "dataset_metadata" / "bonefract_single_source_manifest.csv"
ARCHIVE_URL = "https://data.mendeley.com/public-files/datasets/4cr7f2359x/files/76c032f0-b8be-49b8-a943-d30d77ac62e4/file_downloaded"
SEED = 2026
IMAGE_SIZE = 224
DOWNLOAD_RETRIES = 8
SPLIT_MAP = {"train": "train", "valid": "validation", "test": "test"}


def standardize(raw):
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        image = ImageOps.exif_transpose(image).convert("L")
        image = ImageOps.autocontrast(image)
        image = ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, "JPEG", quality=95, optimize=True)
        return output.getvalue()


def choose_members(names):
    groups = {}
    for name in names:
        path = PurePosixPath(name)
        parts = path.parts
        if len(parts) < 6 or path.suffix.lower() != ".png":
            continue
        split, anatomy, patient, label = parts[1], parts[2], parts[3], parts[4]
        if split not in SPLIT_MAP or anatomy not in {"Hand", "wrist"}:
            continue
        if label not in {"Positive", "Negative"}:
            continue
        groups.setdefault((split, anatomy, label), []).append((name, patient))

    rng = random.Random(SEED)
    selected = []
    for split in SPLIT_MAP:
        for anatomy in ("Hand", "wrist"):
            positives = groups[(split, anatomy, "Positive")]
            negatives = groups[(split, anatomy, "Negative")]
            rng.shuffle(negatives)
            if len(negatives) < len(positives):
                raise RuntimeError(f"Insufficient negatives for {split}/{anatomy}")
            selected.extend(
                (member, patient, split, anatomy, label)
                for label, items in (("Positive", positives), ("Negative", negatives[:len(positives)]))
                for member, patient in items
            )
    rng.shuffle(selected)
    return selected


def read_with_retries(archive, member):
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            return archive.read(member), archive
        except Exception as error:
            archive.close()
            if attempt == DOWNLOAD_RETRIES:
                raise
            delay = min(60, attempt * 5)
            print(
                f"Download interrupted ({type(error).__name__}); reconnecting in {delay}s "
                f"[{attempt}/{DOWNLOAD_RETRIES}]",
                flush=True,
            )
            time.sleep(delay)
            archive = RemoteZip(ARCHIVE_URL)


def main():
    for split in SPLIT_MAP.values():
        for label in ("fractured", "non_fractured"):
            (OUTPUT / split / label).mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(exist_ok=True)

    records = []
    hashes = set()
    archive = RemoteZip(ARCHIVE_URL)
    try:
        selected = choose_members(archive.namelist())
        for number, (member, patient, source_split, anatomy, source_label) in enumerate(selected, 1):
            split = SPLIT_MAP[source_split]
            label = "fractured" if source_label == "Positive" else "non_fractured"
            filename = f"bonefract_{anatomy.lower()}_{patient}.jpg"
            destination = OUTPUT / split / label / filename
            if destination.is_file():
                converted = destination.read_bytes()
            else:
                raw, archive = read_with_retries(archive, member)
                converted = standardize(raw)
            digest = hashlib.sha256(converted).hexdigest()
            if digest in hashes:
                if destination.is_file():
                    destination.unlink()
                print(f"Skipped duplicate: {member}", flush=True)
                continue
            if not destination.is_file():
                destination.write_bytes(converted)
            hashes.add(digest)
            records.append((split, label, anatomy.lower(), patient, filename, member, digest))
            if number % 100 == 0 or number == len(selected):
                print(f"Prepared {number}/{len(selected)}", flush=True)
    finally:
        archive.close()

    patient_splits = {}
    for split, _, _, patient, *_ in records:
        previous = patient_splits.setdefault(patient, split)
        if previous != split:
            raise RuntimeError(f"Patient leakage detected: {patient}")
    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "label", "anatomy", "patient_id", "filename", "source_member", "sha256"])
        writer.writerows(records)
    print(f"Completed {len(records)} images; manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
