# Evaluation

SketchNet is evaluated on artists not seen during training. Images are grouped by the artist prefix in each filename and assigned with a seeded, stratified five-fold split: three folds for training, one for validation, and one for testing.

The best checkpoint is selected using validation balanced accuracy. The test split is then evaluated once. Alongside ordinary accuracy, the report includes balanced accuracy, per-class precision and recall, a confusion matrix, and a 95% confidence interval obtained by resampling whole artists.

## Split

| Split | Images | Artists | Paintings | Sketches |
|---|---:|---:|---:|---:|
| Train | 2,012 | 165 | 1,461 | 551 |
| Validation | 667 | 57 | 486 | 181 |
| Test | 664 | 54 | 481 | 183 |

No artist crosses between splits. Four perceptually similar pairs were reviewed by the audit; every pair shares an artist and class and therefore remains within one split.

## Test result

The selected epoch-4 checkpoint correctly classified 614 of 664 test images: **92.47% accuracy** and **89.56% balanced accuracy**. Artist-level bootstrap intervals are 88.65–95.97% for accuracy and 83.12–93.60% for balanced accuracy.

The confusion matrix uses true classes as rows and predicted classes as columns:

|  | Predicted painting | Predicted sketch |
|---|---:|---:|
| Painting | 462 | 19 |
| Sketch | 31 | 152 |

Painting precision/recall/F1 are 93.71% / 96.05% / 94.87%. Sketch precision/recall/F1 are 88.89% / 83.06% / 85.88%.

The full machine-readable result, including the exact model and split-manifest fingerprints, is stored in [`reports/sketchnet-evaluation.json`](../reports/sketchnet-evaluation.json).

The private mixed-source image collection is not distributed. The committed split manifest and evaluation JSON record the exact inputs and configuration used for the published model.
