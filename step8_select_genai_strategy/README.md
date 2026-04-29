# Step 8 Select GenAI Labeling Strategy

This folder contains the model-comparison and train/test labeling workflow.

## Deliverables

- `notebook/step8_select_genai_strategy_colab.ipynb`: Google Colab notebook that compares Step 7 model labels against Step 6 human consensus, selects a labeling strategy, and labels the Step 5 train/test data with the selected model.
- `scripts/select_genai_strategy_local.py`: local script version of the Step 8 workflow.
- `data/step8_benchmark_results.csv`: benchmark metrics comparing GenAI labels to human consensus.
- `data/selected_labeling_strategy.txt`: selected model and selection metric.
- `data/confusion_matrix_*.png`: confusion matrix charts by model.
- `data/train_labeled_*.csv`, `data/test_labeled_*.csv`, `data/train_test_labeled_*.csv`: labeled train/test outputs from the selected model.

## Run in Colab

Open `notebook/step8_select_genai_strategy_colab.ipynb` in Google Colab.

The notebook reads Step 6 consensus labels, Step 7 model labels, and Step 5 train/test files from Google Drive.

## Run Locally

From the project root:

```bash
python step8_select_genai_strategy/scripts/select_genai_strategy_local.py
```

By default, the local script reads:

- `step6_human_labeling/data/holdout_with_consensus.csv`
- `step7_genai_benchmark/data/*_labels.csv`
- `step7_genai_benchmark/benchmark_summary.csv`
- `step5_preprocess_split/data/train.csv`
- `step5_preprocess_split/data/test.csv`
- `step5_preprocess_split/data/train_test_15000.csv`

It writes Step 8 outputs to `step8_select_genai_strategy/data/`.

To label train/test with the selected model, pass `--label-train-test`. This uses paid provider APIs and requires the matching API key in the environment or `.env`:

```bash
python step8_select_genai_strategy/scripts/select_genai_strategy_local.py --label-train-test
```
