# Data Collection

This folder collects sentence-level text from public SEC 10-K annual reports. The goal is to build a dataset for labeling sentences as exploration, exploitation, ambidextrous, or neither.

The main source text is the MD&A section of each filing. MD&A stands for "Management's Discussion and Analysis" and is usually Item 7 in a 10-K. This section is useful because companies describe strategy, operations, risks, investments, market conditions, and performance in plain language.

## Process

The collection script starts with a list of public company tickers across eight industries. For each ticker, it asks SEC EDGAR for the company's CIK, which is the SEC's company identifier. It then uses the SEC submissions API to find recent 10-K filings and the official primary document for each filing.

After the script finds a filing document, it downloads the HTML from SEC EDGAR, removes tables and other non-sentence content, extracts the MD&A section, and splits the section into individual sentences. Very short, very long, mostly numeric, or boilerplate sentences are filtered out.

The script saves progress after each filing. If it stops or crashes, you can run it again and it will continue from the filings already saved in `data/sentences_raw.csv`.

## Scripts

`scripts/collect_data.py` runs the full collection process and writes `data/sentences_raw.csv`.

`scripts/sec_connection.py` handles SEC EDGAR requests, including ticker-to-CIK lookup, 10-K filing lookup, and filing document downloads.

`scripts/data_cleaning.py` handles text cleanup, MD&A extraction, and sentence filtering.

`scripts/sample_holdout.py` creates the human-labeling holdout set after raw sentence collection is done.

## Setup

From the project root, create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate
pip install requests beautifulsoup4 lxml nltk tqdm pandas
```

SEC requires automated tools to identify themselves. In `scripts/collect_data.py`, the request header should include a real name and email:

```python
"User-Agent": "Your Name your_email@unc.edu",
```

## Collect Sentences

Run this from the `data-collection` folder:

```bash
python scripts/collect_data.py
```

The target is about 16,500 raw sentences. The script also caps how many sentences can come from one company so the dataset is not dominated by a few large filings.

The output is `data/sentences_raw.csv` with these columns:

```text
sentence_id, sentence, ticker, cik, filing_date, filing_id
```

## Create Holdout Set

After `data/sentences_raw.csv` exists, run:

```bash
python scripts/sample_holdout.py
```

This samples exactly 1,000 sentences for human labeling. It tries to spread the sample evenly across industries, companies within each industry, and filing years within each company.

It creates three files:

`data/holdout_1000.csv` contains the 1,000 sentences for human labeling.

`data/train_test_remaining.csv` contains all remaining sentences for model training and testing.

`data/holdout_summary.txt` summarizes the holdout split by industry, company, and filing year.