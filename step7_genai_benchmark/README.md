# Step 7 GenAI Benchmark

This folder contains the GenAI labeling benchmark deliverables.

## Deliverables

- `scripts/benchmark_labels.py`: runs the fixed prompt across six configured models from OpenAI, Anthropic, and Google.
- `data/*_labels.csv`: per-model label outputs.
- `benchmark_summary.csv`: runtime, latency, estimated cost, and label count summary.
- `step7_benchmark_evaluation.ipynb`: notebook comparing model labels against human consensus.
- `step7_benchmark_results.csv`: benchmark metrics table.
- `confusion_matrix_*.png`: model confusion matrix charts.

## Run Model Labeling

From the project root:

```bash
python step7_genai_benchmark/scripts/benchmark_labels.py
```

By default, the script reads `step6_human_labeling/data/holdout_with_consensus.csv` and writes outputs to `step7_genai_benchmark/`. API keys are read from environment variables or `.env`.

## Evaluate Results

Open and run `step7_benchmark_evaluation.ipynb`. The notebook reports agreement against human labels, macro and per-class F1, MCC, and available summary/cost timing fields.
