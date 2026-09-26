# AI-Based Decision Support System for Hand Fractures

This repository contains the cleaned hand/wrist X-ray dataset, final MobileNetV3-Large training pipeline, trained checkpoint, evaluation metrics, graphs, and final progress presentation.

## Final dataset

- Total: 1,690 X-rays
- Fractured: 510
- Non-fractured: 1,180
- Training: 1,182 images
- Validation: 254 images
- Test: 254 images

The prepared images are stored under `hand_dataset_prepared/`.

## Final model

- Architecture: MobileNetV3-Large
- Transfer learning with the last six feature blocks fine-tuned
- Balanced sampling and training-time augmentation
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

Training is CPU-compatible but will run faster with suitable accelerated PyTorch hardware.

## Main files

- `train_advanced.py` — final training and evaluation pipeline
- `model_output_advanced/` — best model, history, and test metrics
- `graph_outputs/` — accuracy, loss, confusion-matrix, and comparison graphs
- `presentation_output/Hand_Fracture_Advanced_Model_With_Graphs_Final.pptx` — final presentation
- `EXPANDED_DATA_EXPERIMENT_REPORT.md` — record of additional-data experiments

## Important limitation

This is an academic decision-support prototype, not a clinically validated diagnostic system or replacement for a radiologist.
