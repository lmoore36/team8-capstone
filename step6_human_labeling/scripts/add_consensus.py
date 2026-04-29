from collections import Counter
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INPUT_FILE = DATA_DIR / "complete-labeled-holdout.csv"
OUTPUT_FILE = DATA_DIR / "holdout_with_consensus.csv"

SENTIMENT_COLUMNS = [
    "reviewer 1 sentiment",
    "reviewer 2 sentiment",
    "reviewer 3 sentiment",
]


def get_consensus(row):
    votes = Counter(row[SENTIMENT_COLUMNS])
    label, count = votes.most_common(1)[0]

    if count >= 2:
        return label

    return "disagreement"


def get_agreement(row):
    votes = Counter(row[SENTIMENT_COLUMNS])
    highest_count = votes.most_common(1)[0][1]

    if highest_count == 3:
        return "unanimous"
    if highest_count == 2:
        return "majority"

    return "disagreement"


def main():
    df = pd.read_csv(INPUT_FILE)

    df["consensus"] = df.apply(get_consensus, axis=1)
    df["agreement"] = df.apply(get_agreement, axis=1)

    print("Consensus counts:")
    print(df["consensus"].value_counts())

    print("\nAgreement counts:")
    print(df["agreement"].value_counts())

    print("\nAgreement percentage breakdown:")
    print(df["agreement"].value_counts(normalize=True).mul(100).round(2))

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved output to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
