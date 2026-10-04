# AI-Based Decision Support System for Hand Fractures

This repository contains the cleaned hand/wrist X-ray dataset, final MobileNetV3-Large training pipeline, trained checkpoint, evaluation metrics, and graphs.

## Final dataset

- Total: 3,046 X-rays
- Fractured: 1,195
- Non-fractured: 1,851
- Training: 2,538 images
- Validation: 254 images
- Test: 254 images

The prepared images are stored under `hand_dataset_prepared/`.

## Final model

- Architecture: MobileNetV3-Large
- Transfer learning with conservative fine-tuning of the final feature blocks
- Class-balanced, source-aware sampling and training-time augmentation
- Best test accuracy: **87.80%**
- Fracture precision: **87.10%**
- Fracture recall: **70.13%**
- Specificity: **95.48%**
- F1-score: **77.70%**

The best checkpoint and its recorded results are in `model_output_advanced/`.

## Run the project

1. Install Python 3.11 or newer.
2. Double-click `setup.bat` once.
3. Double-click `train_advanced.bat` to retrain the model.
4. Double-click `generate_graphs.bat` to regenerate the result graphs.

## Predict one X-ray

Run the command-line predictor:

```powershell
.\.venv\Scripts\python.exe .\predict.py "C:\path\to\xray.png"
```

For the validation-selected screening mode, which detects more fractures at the
cost of more false alarms and lower overall accuracy:

```powershell
.\.venv\Scripts\python.exe .\predict.py "C:\path\to\xray.png" --screening
```

## Test unseen X-rays

Place new X-rays that were not used for training, validation, or testing inside
`unseen_xrays`, then double-click `test_unseen.bat`. Predictions are saved in
`unseen_predictions.csv`. Verify them against reliable labels from the dataset
or a qualified reviewer.

To batch-test with the higher-recall screening threshold, run:

```powershell
.\test_unseen.bat --screening
```

Training is CPU-compatible but will run faster with suitable accelerated PyTorch hardware.

## Main files

- `train_advanced.py` — final training and evaluation pipeline
- `predict.py` — reusable single-image prediction code
- `test_unseen.py` — batch testing for genuinely unseen X-rays
- `model_output_advanced/` — best model, history, and test metrics
- `graph_outputs/` — accuracy, loss, and confusion-matrix graphs

## Important limitation

This is an academic decision-support prototype, not a clinically validated diagnostic system or replacement for a radiologist.
