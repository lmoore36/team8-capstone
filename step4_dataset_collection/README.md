# Step 4 Dataset Collection

This folder contains the collection code and raw dataset deliverable.

## Deliverables

- `data/sentences_raw.csv`: raw sentence-level dataset collected from SEC 10-K MD&A sections.
- `scripts/collect_data.py`: main collection script.
- `scripts/sec_connection.py`: SEC EDGAR request helpers.
- `scripts/data_cleaning.py`: MD&A extraction and sentence filtering helpers.

## Run

From the project root:

```bash
python step4_dataset_collection/scripts/collect_data.py
```

The script targets about 16,500 raw sentences, giving enough data for a 15,000-row train/test pool and a 1,000-row locked holdout. It saves progress to `step4_dataset_collection/data/sentences_raw.csv` after each filing so interrupted runs can resume.

## Collection Rules

SEC requests include a `User-Agent`, use public endpoints, and pause between requests. The script filters out very short, very long, mostly numeric, and obvious boilerplate sentences before writing the raw dataset.
