from pathlib import Path

import pandas as pd


RANDOM_SEED = 42
HOLDOUT_SIZE = 1000
TRAIN_TEST_SIZE = 15000
TEST_FRACTION = 0.20

STEP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = STEP_DIR / "data"
INPUT_FILE = PROJECT_ROOT / "step4_dataset_collection" / "data" / "sentences_raw.csv"
LOCKED_HOLDOUT_FILE = DATA_DIR / "original_holdout_1000.csv"
HOLDOUT_FILE = DATA_DIR / "holdout_1000.csv"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
TRAIN_TEST_FILE = DATA_DIR / "train_test_15000.csv"
SUMMARY_FILE = DATA_DIR / "preprocess_split_summary.txt"

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


def add_filing_year(df):
    df = df.copy()
    df["filing_year"] = pd.to_datetime(df["filing_date"], errors="coerce").dt.year.astype("Int64").astype(str)
    return df


def load_sentences():
    df = pd.read_csv(INPUT_FILE, dtype=str)
    df = df.dropna(subset=["sentence"])
    df["sentence"] = df["sentence"].str.strip()
    df = df[df["sentence"].ne("")]
    df = df.drop_duplicates(subset=["sentence"]).copy()
    df["ticker"] = df["ticker"].str.upper()
    df["industry"] = df["ticker"].map(TICKER_TO_INDUSTRY).fillna("Unknown")
    df = add_filing_year(df)
    return df[df["industry"] != "Unknown"].copy()


def load_locked_holdout():
    """Read the hand-labeled holdout source without modifying it."""
    if not LOCKED_HOLDOUT_FILE.exists():
        return None

    holdout = pd.read_csv(LOCKED_HOLDOUT_FILE, dtype=str)
    missing_columns = sorted(set(OUTPUT_COLUMNS) - set(holdout.columns))
    if missing_columns:
        raise ValueError(
            f"{LOCKED_HOLDOUT_FILE} is missing required columns: {', '.join(missing_columns)}"
        )

    holdout = holdout[OUTPUT_COLUMNS].dropna(subset=["sentence"]).copy()
    holdout["sentence"] = holdout["sentence"].str.strip()
    holdout = holdout[holdout["sentence"].ne("")]

    if len(holdout) != HOLDOUT_SIZE:
        raise ValueError(
            f"{LOCKED_HOLDOUT_FILE} should contain {HOLDOUT_SIZE} rows, found {len(holdout)}"
        )
    if holdout["sentence_id"].duplicated().any():
        raise ValueError(f"{LOCKED_HOLDOUT_FILE} contains duplicate sentence_id values")
    if holdout["sentence"].duplicated().any():
        raise ValueError(f"{LOCKED_HOLDOUT_FILE} contains duplicate sentence values")

    return add_filing_year(holdout)


def remove_holdout_from_pool(df, holdout):
    remaining = df.copy()

    if "sentence_id" in remaining.columns and "sentence_id" in holdout.columns:
        remaining = remaining[~remaining["sentence_id"].isin(holdout["sentence_id"])]

    return remaining[~remaining["sentence"].isin(holdout["sentence"])].copy()


def write_generated_csv(df, output_file):
    if output_file.resolve() == LOCKED_HOLDOUT_FILE.resolve():
        raise ValueError(f"Refusing to overwrite locked holdout source: {LOCKED_HOLDOUT_FILE}")

    df[OUTPUT_COLUMNS].to_csv(output_file, index=False)


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


def sample_pool(df, target):
    """Sample up to target rows across industry, ticker, and filing year."""
    if len(df) <= target:
        return df.copy()

    parts = []
    industry_targets = allocate_targets(df.groupby("industry").size(), target)
    for industry, industry_target in industry_targets.items():
        industry_df = df[df["industry"] == industry]
        parts.append(sample_evenly(industry_df, ["ticker", "filing_year"], industry_target))

    sampled = pd.concat(parts).drop_duplicates(subset=["sentence"])
    if len(sampled) > target:
        sampled = sampled.sample(n=target, random_state=RANDOM_SEED)
    return sampled


def create_sampled_holdout(df):
    holdout = sample_pool(df, HOLDOUT_SIZE)

    remaining_needed = min(HOLDOUT_SIZE, len(df)) - len(holdout)
    if remaining_needed > 0:
        remaining_pool = df.drop(index=holdout.index)
        holdout = pd.concat([
            holdout,
            sample_evenly(remaining_pool, ["industry", "ticker", "filing_year"], remaining_needed),
        ])

    return holdout


def get_holdout(df):
    locked_holdout = load_locked_holdout()
    if locked_holdout is not None:
        return locked_holdout

    return create_sampled_holdout(df)


def split_train_test(train_test):
    test_size = round(len(train_test) * TEST_FRACTION)
    test = sample_pool(train_test, test_size)
    train = train_test.drop(index=test.index)
    return train, test


def length_stats(df):
    lengths = df["sentence"].str.split().str.len()
    return {
        "rows": len(df),
        "mean_words": round(lengths.mean(), 2),
        "median_words": round(lengths.median(), 2),
        "min_words": int(lengths.min()),
        "max_words": int(lengths.max()),
    }


def format_stats(name, df):
    stats = length_stats(df)
    return (
        f"{name}: rows={stats['rows']}, mean_words={stats['mean_words']}, "
        f"median_words={stats['median_words']}, min_words={stats['min_words']}, "
        f"max_words={stats['max_words']}"
    )


def build_summary(df, holdout, train_test, train, test):
    train_test_shortfall = max(0, TRAIN_TEST_SIZE - len(train_test))
    lines = [
        "Step 5 Preprocess and Split Summary",
        "=" * 36,
        f"Raw cleaned sentences available: {len(df)}",
        f"Locked holdout sentences: {len(holdout)}",
        f"Train/test target sentences: {TRAIN_TEST_SIZE}",
        f"Train/test actual sentences: {len(train_test)}",
        f"Train/test shortfall sentences: {train_test_shortfall}",
        f"Train sentences: {len(train)}",
        f"Test sentences: {len(test)}",
        "",
        "Length statistics:",
        format_stats("All cleaned", df),
        format_stats("Holdout", holdout),
        format_stats("Train", train),
        format_stats("Test", test),
        "",
        "Holdout breakdown by industry:",
        holdout["industry"].value_counts().sort_index().to_string(),
        "",
        "Train/test breakdown by industry:",
        train_test["industry"].value_counts().sort_index().to_string(),
        "",
        "Holdout breakdown by company:",
        holdout["ticker"].value_counts().sort_index().to_string(),
        "",
        "Holdout breakdown by filing year:",
        holdout["filing_year"].value_counts().sort_index().to_string(),
        "",
        "Imbalance plan:",
        "The holdout is fixed from the hand-labeled original so labels stay reproducible. The train/test files are sampled across industry, ticker, and filing year to reduce firm concentration. After human labels are available, report label distributions and use macro F1/MCC plus class weighting or balanced sampling if one strategic-orientation class is rare.",
        "",
    ]
    return "\n".join(lines)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_sentences()
    holdout = get_holdout(df)
    remaining = remove_holdout_from_pool(df, holdout)
    train_test = sample_pool(remaining, TRAIN_TEST_SIZE)
    train, test = split_train_test(train_test)

    write_generated_csv(holdout, HOLDOUT_FILE)
    write_generated_csv(train_test, TRAIN_TEST_FILE)
    write_generated_csv(train, TRAIN_FILE)
    write_generated_csv(test, TEST_FILE)

    summary = build_summary(df, holdout, train_test, train, test)
    with SUMMARY_FILE.open("w", encoding="utf-8") as summary_file:
        summary_file.write(summary)

    print(summary)
    print(f"Saved holdout to: {HOLDOUT_FILE}")
    print(f"Saved train/test set to: {TRAIN_TEST_FILE}")
    print(f"Saved train split to: {TRAIN_FILE}")
    print(f"Saved test split to: {TEST_FILE}")
    print(f"Saved summary to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
