# Dataset source

The final dataset was prepared from **FracAtlas**, using only hand and wrist X-rays relevant to binary fracture classification.

Images that were elbow-only, forearm/upper-arm, unrelated to the hand or wrist, unclear, or corrupted were excluded during manual review.

## Final split

| Split | Fractured | Non-fractured | Total |
|---|---:|---:|---:|
| Training | 356 | 826 | 1,182 |
| Validation | 77 | 177 | 254 |
| Test | 77 | 177 | 254 |
| **Total** | **510** | **1,180** | **1,690** |

Training imbalance is handled with weighted sampling. Validation and test images are never augmented or used to update model weights.
