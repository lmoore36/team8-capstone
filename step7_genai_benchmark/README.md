# Step 7 GenAI Benchmark

This folder contains the GenAI labeling benchmark deliverables.

## Deliverables

- `scripts/benchmark_labels.py`: runs the fixed prompt across six configured models from OpenAI, Anthropic, and Google.
- `data/*_labels.csv`: per-model label outputs.
- `benchmark_summary.csv`: runtime, latency, estimated cost, and label count summary.

## Run Model Labeling

From the project root:

```bash
python step7_genai_benchmark/scripts/benchmark_labels.py
```

By default, the script reads `step6_human_labeling/data/holdout_with_consensus.csv` and writes outputs to `step7_genai_benchmark/`. API keys are read from environment variables or `.env`.

## Next Step

Use Step 8 to compare model labels against human consensus, select a GenAI labeling strategy, and label the train/test data.
