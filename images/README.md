# Dataset location (not bundled)

The training/test JPEG files and the UAB split/ground-truth JSON files are intentionally **not** distributed with this repository.

To reproduce the original course experiments, obtain the corresponding materials through an authorized source and place them at:

```text
images/
  gt.json
  gt_reduced.json
  train/<image_id>.jpg
  test/<image_id>.jpg
```

The public Kaggle Fashion Product Images dataset is the upstream dataset named in the assignment, but the **course-specific split and JSON ground truth** are required by `my_labeling.py`. Simply downloading Kaggle files is not enough.
