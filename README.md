# AI-Based Decision Support System for Hand Fractures

This project classifies hand and wrist X-rays as `fractured` or `non_fractured` using a transfer-learned MobileNetV3-Large model.

## Final dataset

- Total: 1,690 X-rays
- Training: 356 fractured and 826 non-fractured
- Validation: 77 fractured and 177 non-fractured
- Test: 77 fractured and 177 non-fractured

The images are stored in `hand_dataset/`. During training, weighted sampling presents both classes equally often without duplicating the saved image files. Validation and test data remain unchanged so evaluation reflects the real held-out distribution.

## Final result

- Test accuracy: **87.80%**
- Fracture precision: **87.10%**
- Fracture recall: **70.13%**
- Specificity: **95.48%**
- Fracture F1-score: **77.70%**
- Confusion matrix: `[[54, 23], [8, 169]]`

The selected checkpoint and recorded results are in `model_output/`. Graphs are in `graph_outputs/`.

## Run the project

1. Install Python 3.11 or newer.
2. Run `setup.bat` once.
3. Run `train.bat` to train and evaluate the model.
4. Run `generate_graphs.bat` to regenerate the graphs from the latest saved results.

```powershell
.\setup.bat
.\train.bat
.\generate_graphs.bat
```

## Predict one X-ray

```powershell
.\.venv\Scripts\python.exe .\predict.py "C:\path\to\xray.png"
```

Add `--screening` to use the validation-selected higher-recall threshold. It can detect more fractures but may produce more false alarms.

## Test unseen X-rays

Put genuinely unseen images inside `unseen_xrays/`, then run `test_unseen.bat`. Predictions are saved in `unseen_predictions.csv`.

## Main files

- `train.py` — training, validation, model selection, and final testing
- `predict.py` — single-image prediction
- `test_unseen.py` — batch prediction on unseen X-rays
- `tune_threshold.py` — validation-based threshold selection
- `generate_graphs.py` — result graphs
- `hand_dataset/` — final dataset
- `model_output/` — checkpoint, history, threshold, and test metrics

## Limitation

This is an academic decision-support prototype. It is not clinically validated and must not replace assessment by a qualified medical professional.
