import argparse
import json
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms


PROJECT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT / "model_output_advanced" / "best_advanced_model.pt"
THRESHOLD_REPORT = PROJECT / "model_output_advanced" / "decision_threshold.json"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_model(checkpoint_path=DEFAULT_MODEL):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "mobilenet_v3_large":
        raise ValueError("This predictor expects a MobileNetV3-Large checkpoint.")

    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    class_to_idx = checkpoint.get("class_to_idx", {"fractured": 0, "non_fractured": 1})
    image_size = int(checkpoint.get("image_size", 224))
    return model, class_to_idx, image_size


class FracturePredictor:
    def __init__(self, checkpoint_path=DEFAULT_MODEL, fracture_threshold=0.5):
        self.model, self.class_to_idx, self.image_size = build_model(checkpoint_path)
        self.fracture_threshold = float(fracture_threshold)
        self.idx_to_class = {index: name for name, index in self.class_to_idx.items()}
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path):
        image_path = Path(image_path)
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {image_path.suffix or 'none'}")
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError("The selected file is not a valid readable image.") from error

        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]

        fractured_index = self.class_to_idx["fractured"]
        normal_index = self.class_to_idx["non_fractured"]
        predicted_index = (
            fractured_index
            if float(probabilities[fractured_index]) >= self.fracture_threshold
            else normal_index
        )
        predicted_class = self.idx_to_class[predicted_index]
        class_probabilities = {
            name: float(probabilities[index].item())
            for name, index in self.class_to_idx.items()
        }
        return {
            "image": str(image_path.resolve()),
            "prediction": predicted_class,
            "confidence": class_probabilities[predicted_class],
            "probabilities": class_probabilities,
            "fracture_threshold": self.fracture_threshold,
        }


def main():
    parser = argparse.ArgumentParser(description="Predict one hand or wrist X-ray.")
    parser.add_argument("image", help="Path to the X-ray image")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Checkpoint path")
    parser.add_argument(
        "--screening",
        action="store_true",
        help="Use the validation-selected high-recall fracture threshold",
    )
    args = parser.parse_args()
    threshold = 0.5
    if args.screening:
        if not THRESHOLD_REPORT.is_file():
            raise FileNotFoundError(f"Threshold report not found: {THRESHOLD_REPORT}")
        threshold = json.loads(THRESHOLD_REPORT.read_text(encoding="utf-8"))["fracture_threshold"]
    print(json.dumps(FracturePredictor(args.model, threshold).predict(args.image), indent=2))


if __name__ == "__main__":
    main()
