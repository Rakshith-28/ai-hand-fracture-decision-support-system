# Dataset sources

## BoneFract additions

- Dataset: **BoneFract: A Bone Fracture Dataset**
- Official record: https://doi.org/10.17632/4cr7f2359x.1
- License: CC BY 4.0
- Positive use: 350 Hand-positive and 350 Wrist-positive candidates selected with seed 42; 685 unique images remained after duplicate removal
- Negative use: matched Hand-negative and Wrist-negative candidates selected with seed 43; 671 unique images remained after duplicate removal
- Patient handling: the source provides one original X-ray per unique patient
- Processing: grayscale conversion, automatic contrast, center crop, and resize to 224×224
- Labels: anatomy and fracture status supplied by the dataset authors
- Limitation: the public metadata does not include patient ages, so this subset cannot be certified as adult-only

The supplied validation and test sets were not modified, so the existing held-out evaluation remains comparable. Positive and negative manifests are stored in `dataset_metadata/bonefract_added_manifest.csv` and `dataset_metadata/bonefract_negative_manifest.csv`.

## BoneFract single-source experiment

The separate `bonefract_single_source/` dataset preserves BoneFract's supplied
train, validation, and test divisions. Within every division and anatomy, an
equal number of positive and negative candidates was selected. Content-level
duplicates were removed, leaving 5,378 unique images. The complete provenance,
patient identifiers, source paths, and SHA-256 hashes are recorded in
`dataset_metadata/bonefract_single_source_manifest.csv`.
