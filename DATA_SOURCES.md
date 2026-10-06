# Dataset source

The dataset combines the reviewed **FracAtlas** hand/wrist collection with 470 additional original fracture-positive training X-rays from **BoneFract: A Bone Fracture Dataset** (DOI: 10.17632/4cr7f2359x.1, CC BY 4.0).

Images that were elbow-only, forearm/upper-arm, unrelated to the hand or wrist, unclear, or corrupted were excluded during manual review.

## Final split

| Split | Fractured | Non-fractured | Total |
|---|---:|---:|---:|
| Training | 826 | 826 | 1,652 |
| Validation | 177 | 177 | 354 |
| Test | 177 | 177 | 354 |
| **Total** | **1,180** | **1,180** | **2,360** |

The first 470 additions contain 235 Hand-positive and 235 Wrist-positive images from BoneFract's training partition. A second verified import balanced validation and test at 177 images per class and replaced duplicate files found during the full-dataset SHA-256 audit. The source states that each image represents a unique patient and that the release contains original images rather than augmented copies. No training augmentation is enabled. Validation and test images are never used to update model weights. Import details are recorded in `additional_fractured_manifest.csv` and `evaluation_balance_manifest.csv`.
