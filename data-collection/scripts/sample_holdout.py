import os

import pandas as pd


RANDOM_SEED = 42
HOLDOUT_SIZE = 1000

DATA_COLLECTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(DATA_COLLECTION_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "sentences_raw.csv")
LEGACY_INPUT_FILE = os.path.join(DATA_COLLECTION_DIR, "sentences_raw.csv")
HOLDOUT_FILE = os.path.join(DATA_DIR, "holdout_1000.csv")
TRAIN_TEST_FILE = os.path.join(DATA_DIR, "train_test_remaining.csv")
SUMMARY_FILE = os.path.join(DATA_DIR, "holdout_summary.txt")

INDUSTRY_TICKERS = {
    "Tech": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "INTC", "IBM", "ORCL", "CSCO", "ADBE"],
    "Healthcare": ["JNJ", "PFE", "MRK", "ABT", "BMY", "AMGN", "GILD", "MDT", "UNH", "CVS"],
    "Finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "COF", "USB"],
    "Consumer/Retail": ["WMT", "AMZN", "TGT", "COST", "HD", "LOW", "NKE", "SBUX", "MCD", "YUM"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "PSX", "VLO", "MPC", "OXY", "HES", "DVN"],
    "Industrials": ["GE", "HON", "MMM", "CAT", "DE", "BA", "LMT", "RTX", "UPS", "FDX"],
    "Telecom/Media": ["T", "VZ", "CMCSA", "DIS", "NFLX", "PARA", "WBD", "FOXA", "DISH", "LUMN"],
    "Materials": ["DD", "DOW", "LIN", "APD", "NEM", "FCX", "VMC", "MLM", "PKG", "IP"],
}

TICKER_TO_INDUSTRY = {
    ticker: industry
    for industry, tickers in INDUSTRY_TICKERS.items()
    for ticker in tickers
}

OUTPUT_COLUMNS = ["sentence_id", "sentence", "ticker", "industry", "filing_date", "filing_id"]


def load_sentences():
    input_file = INPUT_FILE if os.path.exists(INPUT_FILE) else LEGACY_INPUT_FILE
    df = pd.read_csv(input_file, dtype=str)
    df = df.drop_duplicates(subset=["sentence"]).copy()
    df["ticker"] = df["ticker"].str.upper()
    df["industry"] = df["ticker"].map(TICKER_TO_INDUSTRY).fillna("Unknown")
    df["filing_year"] = pd.to_datetime(df["filing_date"], errors="coerce").dt.year.astype("Int64").astype(str)
    return df[df["industry"] != "Unknown"].copy()


def allocate_targets(group_sizes, total_target):
    """Allocate target rows across groups, redistributing shortages to groups with capacity."""
    groups = list(group_sizes.index)
    if not groups:
        return {}

    base = total_target // len(groups)
    remainder = total_target % len(groups)
    targets = {
        group: min(int(group_sizes[group]), base + (1 if i < remainder else 0))
        for i, group in enumerate(groups)
    }

    while sum(targets.values()) < min(total_target, int(group_sizes.sum())):
        available = {
            group: int(group_sizes[group]) - targets[group]
            for group in groups
            if targets[group] < int(group_sizes[group])
        }
        if not available:
            break

        total_available = sum(available.values())
        remaining = min(total_target, int(group_sizes.sum())) - sum(targets.values())

        for group in sorted(available, key=available.get, reverse=True):
            if remaining <= 0:
                break
            add = max(1, round(remaining * available[group] / total_available))
            add = min(add, available[group], remaining)
            targets[group] += add
            remaining -= add

    return targets


def sample_evenly(df, group_columns, target):
    """Sample target rows while spreading picks evenly across nested groups."""
    if target <= 0 or df.empty:
        return df.head(0)
    if len(df) <= target:
        return df.copy()
    if not group_columns:
        return df.sample(n=target, random_state=RANDOM_SEED)

    group_sizes = df.groupby(group_columns, dropna=False).size()
    targets = allocate_targets(group_sizes, target)

    samples = []
    for group_key, group_target in targets.items():
        if group_target <= 0:
            continue
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        mask = pd.Series(True, index=df.index)
        for column, value in zip(group_columns, group_key):
            mask &= df[column].eq(value)

        group_df = df[mask]
        samples.append(sample_evenly(group_df, group_columns[1:], group_target))

    sampled = pd.concat(samples) if samples else df.head(0)
    if len(sampled) > target:
        sampled = sampled.sample(n=target, random_state=RANDOM_SEED)
    return sampled


def build_summary(holdout):
    lines = [
        "Holdout Sampling Summary",
        "=" * 24,
        f"Total sentences sampled: {len(holdout)}",
        "",
        "Breakdown by industry:",
        holdout["industry"].value_counts().sort_index().to_string(),
        "",
        "Breakdown by company:",
        holdout["ticker"].value_counts().sort_index().to_string(),
        "",
        "Breakdown by filing year:",
        holdout["filing_year"].value_counts().sort_index().to_string(),
        "",
    ]
    return "\n".join(lines)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    df = load_sentences()
    industry_targets = allocate_targets(df.groupby("industry").size(), HOLDOUT_SIZE)

    holdout_parts = []
    for industry, target in industry_targets.items():
        industry_df = df[df["industry"] == industry]
        holdout_parts.append(sample_evenly(industry_df, ["ticker", "filing_year"], target))

    holdout = pd.concat(holdout_parts).drop_duplicates(subset=["sentence"])
    if len(holdout) > HOLDOUT_SIZE:
        holdout = holdout.sample(n=HOLDOUT_SIZE, random_state=RANDOM_SEED)

    remaining_needed = min(HOLDOUT_SIZE, len(df)) - len(holdout)
    if remaining_needed > 0:
        remaining_pool = df.drop(index=holdout.index)
        holdout = pd.concat([
            holdout,
            sample_evenly(remaining_pool, ["industry", "ticker", "filing_year"], remaining_needed),
        ])

    train_test = df.drop(index=holdout.index)

    holdout[OUTPUT_COLUMNS].to_csv(HOLDOUT_FILE, index=False)
    train_test[OUTPUT_COLUMNS].to_csv(TRAIN_TEST_FILE, index=False)

    summary = build_summary(holdout)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as summary_file:
        summary_file.write(summary)

    print(summary)
    print(f"Saved holdout to: {HOLDOUT_FILE}")
    print(f"Saved training/test remainder to: {TRAIN_TEST_FILE}")
    print(f"Saved summary to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
