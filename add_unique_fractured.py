import csv
import hashlib
import io
import random
import zipfile
from pathlib import Path

from PIL import Image, ImageOps


PROJECT = Path(__file__).resolve().parent
ARCHIVE = PROJECT / "raw_data" / "BoneFract.zip"
DESTINATION = PROJECT / "hand_dataset" / "train" / "fractured"
MANIFEST = PROJECT / "additional_fractured_manifest.csv"
PER_ANATOMY = 235
SEED = 42


def standardized_png(raw_bytes):
    with Image.open(io.BytesIO(raw_bytes)) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
        return output.getvalue()


def main():
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"Missing archive: {ARCHIVE}")

    DESTINATION.mkdir(parents=True, exist_ok=True)
    existing_hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in DESTINATION.iterdir()
        if path.is_file()
    }
    rng = random.Random(SEED)
    selected = []

    with zipfile.ZipFile(ARCHIVE) as archive:
        candidates = {"Hand": [], "wrist": []}
        for member in archive.namelist():
            parts = member.split("/")
            if (
                len(parts) == 6
                and parts[1] == "train"
                and parts[2] in candidates
                and parts[4] == "Positive"
                and member.lower().endswith(".png")
            ):
                candidates[parts[2]].append(member)

        for anatomy, members in candidates.items():
            rng.shuffle(members)
            accepted = 0
            for member in members:
                data = standardized_png(archive.read(member))
                digest = hashlib.sha256(data).hexdigest()
                filename = Path(member).name
                output_path = DESTINATION / filename
                if digest in existing_hashes or output_path.exists():
                    continue
                output_path.write_bytes(data)
                existing_hashes.add(digest)
                parts = member.split("/")
                selected.append({
                    "filename": filename,
                    "anatomy": anatomy,
                    "patient_id": parts[3],
                    "label": "fractured",
                    "source_split": "train",
                    "source_member": member,
                    "processed_sha256": digest,
                })
                accepted += 1
                if accepted == PER_ANATOMY:
                    break
            if accepted != PER_ANATOMY:
                raise RuntimeError(f"Only {accepted} unique {anatomy} images were available")

    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=selected[0].keys())
        writer.writeheader()
        writer.writerows(selected)

    print(f"Added {len(selected)} unique fractured X-rays")
    print(f"Hand: {sum(row['anatomy'] == 'Hand' for row in selected)}")
    print(f"Wrist: {sum(row['anatomy'] == 'wrist' for row in selected)}")
    print(f"Training fractured total: {len(list(DESTINATION.iterdir()))}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
