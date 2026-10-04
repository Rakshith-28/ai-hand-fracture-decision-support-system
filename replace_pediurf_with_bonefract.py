import hashlib
import io
import random
import shutil
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps, UnidentifiedImageError
from remotezip import RemoteZip


PROJECT = Path(__file__).resolve().parent
DATASET = PROJECT / "hand_dataset_prepared"
DESTINATION = DATASET / "train" / "fractured"
METADATA = PROJECT / "dataset_metadata"
STAGING = PROJECT / "bonefract_staging"
BACKUP = PROJECT / "pediurf_replacement_backup"
ARCHIVE_URL = "https://data.mendeley.com/public-files/datasets/4cr7f2359x/files/76c032f0-b8be-49b8-a943-d30d77ac62e4/file_downloaded"
SEED = 42
PER_ANATOMY = 350
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


def existing_standardized_hashes():
    hashes = set()
    for path in DATASET.rglob("*"):
        if not path.is_file() or path.name.startswith("pediurf_"):
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            converted = standardized_jpeg(path.read_bytes())
        except (UnidentifiedImageError, OSError):
            continue
        hashes.add(hashlib.sha256(converted).hexdigest())
    return hashes


def select_members(names):
    candidates = {"Hand": [], "wrist": []}
    for name in names:
        path = PurePosixPath(name)
        parts = path.parts
        if len(parts) < 6 or path.suffix.lower() != ".png":
            continue
        if parts[1] != "train" or parts[4] != "Positive":
            continue
        if parts[2] in candidates:
            candidates[parts[2]].append(name)

    rng = random.Random(SEED)
    selected = []
    for anatomy in ("Hand", "wrist"):
        rng.shuffle(candidates[anatomy])
        if len(candidates[anatomy]) < PER_ANATOMY:
            raise RuntimeError(f"Not enough {anatomy} positive images")
        selected.extend(candidates[anatomy][:PER_ANATOMY])
    rng.shuffle(selected)
    return selected


def main():
    if STAGING.exists() or BACKUP.exists():
        raise RuntimeError("A staging or backup folder already exists; inspect it first.")
    STAGING.mkdir(parents=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    existing_hashes = existing_standardized_hashes()
    new_hashes = set()
    records = []

    try:
        with RemoteZip(ARCHIVE_URL) as archive:
            selected = select_members(archive.namelist())
            for number, member in enumerate(selected, start=1):
                converted = standardized_jpeg(archive.read(member))
                digest = hashlib.sha256(converted).hexdigest()
                if digest in existing_hashes or digest in new_hashes:
                    continue
                new_hashes.add(digest)
                source = PurePosixPath(member)
                anatomy = source.parts[2].lower()
                patient = source.parts[3]
                filename = f"bonefract_{anatomy}_{patient}.jpg"
                (STAGING / filename).write_bytes(converted)
                records.append((filename, anatomy, patient, member))
                if number % 25 == 0 or number == len(selected):
                    print(f"Downloaded {number}/{len(selected)} ({len(records)} unique)", flush=True)

        if len(records) < 680:
            raise RuntimeError(f"Only {len(records)} unique images prepared; expected close to 700")

        pediurf_files = list(DESTINATION.glob("pediurf_*.jpg"))
        if len(pediurf_files) != 700:
            raise RuntimeError(f"Expected 700 PediURF files, found {len(pediurf_files)}")
        BACKUP.mkdir()
        for path in pediurf_files:
            shutil.move(str(path), str(BACKUP / path.name))

        try:
            for path in STAGING.iterdir():
                target = DESTINATION / path.name
                if target.exists():
                    raise FileExistsError(target)
                shutil.move(str(path), str(target))
        except Exception:
            for path in DESTINATION.glob("bonefract_*.jpg"):
                path.unlink()
            for path in BACKUP.iterdir():
                shutil.move(str(path), str(DESTINATION / path.name))
            raise

        manifest = METADATA / "bonefract_added_manifest.csv"
        manifest.write_text(
            "filename,anatomy,patient_id,source_member\n"
            + "\n".join(",".join(record) for record in records)
            + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(BACKUP)
        print(f"Replaced 700 PediURF images with {len(records)} BoneFract images")
        print(f"Manifest: {manifest}")
    finally:
        if STAGING.exists():
            shutil.rmtree(STAGING)


if __name__ == "__main__":
    main()
