# Expanded fracture-data experiment

## Data added

- Source: BoneFract: A Bone Fracture Dataset
- DOI: 10.17632/4cr7f2359x.1
- License: CC BY 4.0
- Source partition used: `train` only
- Downloaded subset: 300 fractured hand and 300 fractured wrist X-rays
- Eight normalized duplicates were skipped before selecting the 600 unique images.
- The existing 254-image project test set was kept byte-for-byte unchanged.

## Controlled experiments

| Experiment | Test accuracy | Fracture precision | Fracture recall | F1-score | Confusion matrix |
|---|---:|---:|---:|---:|---|
| Preserved original advanced model | **87.80%** | **87.10%** | 70.13% | **77.70%** | `[[54,23],[8,169]]` |
| 600-image hand+wrist expansion | 86.61% | 84.13% | 68.83% | 75.71% | `[[53,24],[10,167]]` |
| 300-image wrist-only expansion | 85.43% | 77.78% | **72.73%** | 75.17% | `[[56,21],[16,161]]` |
| Validation-selected ensemble | 87.40% | 84.62% | 71.43% | 77.46% | `[[55,22],[10,167]]` |

Rows in each matrix are actual `[fractured, non-fractured]`; columns are predicted `[fractured, non-fractured]`.

## Decision

The original `model_output_advanced/best_advanced_model.pt` remains the best accuracy model. It was not overwritten. The wrist-only experiment detected two more fractures than the original model, but it produced eight additional false fracture alarms, reducing overall accuracy.

The added data are preserved under `additional_images/BoneFract/fractured`. The experimental checkpoints and metrics are preserved under `model_output_expanded`, `model_output_wrist_expanded`, and `model_output_ensemble` for reproducibility.
