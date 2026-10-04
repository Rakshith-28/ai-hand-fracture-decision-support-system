import argparse
import csv
import json
from pathlib import Path

from predict import FracturePredictor, SUPPORTED_EXTENSIONS, THRESHOLD_REPORT


PROJECT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT / "unseen_xrays"
DEFAULT_OUTPUT = PROJECT / "unseen_predictions.csv"


def find_images(folder):
    return sorted(
        path for path in Path(folder).rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main():
    parser = argparse.ArgumentParser(description="Run predictions on unseen X-rays.")
    parser.add_argument("folder", nargs="?", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--screening",
        action="store_true",
        help="Use the validation-selected high-recall fracture threshold",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)
    images = find_images(folder)
    if not images:
        print(f"No X-rays found in: {folder.resolve()}")
        print("Add images there and run this command again.")
        return

    threshold = 0.5
    if args.screening:
        if not THRESHOLD_REPORT.is_file():
            raise FileNotFoundError(f"Threshold report not found: {THRESHOLD_REPORT}")
        threshold = json.loads(THRESHOLD_REPORT.read_text(encoding="utf-8"))["fracture_threshold"]
    predictor = FracturePredictor(fracture_threshold=threshold)
    rows = []
    for image in images:
        try:
            result = predictor.predict(image)
            rows.append({
                "image": str(image.relative_to(folder)),
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "fractured_probability": result["probabilities"]["fractured"],
                "non_fractured_probability": result["probabilities"]["non_fractured"],
                "error": "",
            })
            print(f"{image.name}: {result['prediction']} ({result['confidence'] * 100:.1f}%)")
        except Exception as error:
            rows.append({
                "image": str(image.relative_to(folder)),
                "prediction": "",
                "confidence": "",
                "fractured_probability": "",
                "non_fractured_probability": "",
                "error": str(error),
            })
            print(f"{image.name}: ERROR - {error}")

    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results to: {output.resolve()}")


if __name__ == "__main__":
    main()
