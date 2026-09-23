# PortraitNet evaluation

PortraitNet is evaluated on artists not seen during training. Images are grouped by the artist prefix in each filename and assigned with a seeded, stratified five-fold split: three folds for training, one for validation, and one for testing.

The best checkpoint is selected using validation balanced accuracy. The test split is then evaluated once, with repeat evaluation used only to confirm deterministic output.

## Split

| Split | Images | Artists | General paintings | Portraits |
|---|---:|---:|---:|---:|
| Train | 1,849 | 136 | 1,006 | 843 |
| Validation | 611 | 54 | 334 | 277 |
| Test | 617 | 61 | 337 | 280 |

No artist or confirmed duplicate crosses between splits. Six perceptually similar pairs were flagged; every pair shares an artist and class and therefore remains within one split.

## Test result

The selected epoch-6 checkpoint correctly classified 594 of 617 test images: **96.27% accuracy** and **96.19% balanced accuracy**. Artist-level bootstrap intervals are 93.75–98.04% for accuracy and 92.82–98.03% for balanced accuracy.

The confusion matrix uses true classes as rows and predicted classes as columns:

|  | Predicted general painting | Predicted portrait |
|---|---:|---:|
| General painting | 327 | 10 |
| Portrait | 13 | 267 |

General-painting precision/recall/F1 are 96.18% / 97.03% / 96.60%. Portrait precision/recall/F1 are 96.39% / 95.36% / 95.87%.

The full machine-readable result, including the exact model and split-manifest fingerprints, is stored in [`reports/portraitnet-evaluation.json`](../reports/portraitnet-evaluation.json).

The private mixed-source image collection is not distributed. The historical notebook's random-split validation score is not used as a headline result because artists were not held out.
