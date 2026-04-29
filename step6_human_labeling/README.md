# Step 6 Human Labeling

This folder contains the human-labeling and reliability deliverables.

## Deliverables

- `data/complete-labeled-holdout.csv`: holdout with three human labels per item.
- `scripts/add_consensus.py`: creates consensus and agreement columns from the three reviewer labels.
- `data/holdout_with_consensus.csv`: human labels plus consensus/agreement fields.
- `scripts/step6_reliability.py`: computes Krippendorff's alpha, pairwise Cohen's kappa, and confusion counts.
- `data/step6_reliability_report.csv`: reliability report.

## Run

From the project root:

```bash
python step6_human_labeling/scripts/add_consensus.py
python step6_human_labeling/scripts/step6_reliability.py
```

The reliability script expects reviewer labels to use the normalized classes `neither`, `exploitation`, `exploration`, and `ambiguous`.
