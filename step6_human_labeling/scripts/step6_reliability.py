import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INPUT_FILE = DATA_DIR / "holdout_with_consensus.csv"
OUTPUT_FILE = DATA_DIR / "step6_reliability_report.csv"

SENTIMENT_COLUMNS = [
    "reviewer 1 sentiment",
    "reviewer 2 sentiment",
    "reviewer 3 sentiment",
]

LABELS = ["neither", "exploitation", "exploration", "ambiguous"]
LABEL_TO_CODE = {label: index for index, label in enumerate(LABELS)}
REVIEWER_PAIRS = [
    ("reviewer 1 sentiment", "reviewer 2 sentiment"),
    ("reviewer 1 sentiment", "reviewer 3 sentiment"),
    ("reviewer 2 sentiment", "reviewer 3 sentiment"),
]


def ensure_krippendorff():
    try:
        import krippendorff
    except ImportError:
        print("krippendorff not found. Installing with pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "krippendorff"])
        import krippendorff

    return krippendorff


def normalize_label(label):
    if pd.isna(label):
        return pd.NA

    return str(label).strip().lower()


def build_reliability_matrix(df):
    return df[SENTIMENT_COLUMNS].T.to_numpy()


def build_alpha_matrix(reliability_matrix):
    coded_matrix = []
    for coder_labels in reliability_matrix:
        coded_matrix.append(
            [
                float("nan") if pd.isna(label) else LABEL_TO_CODE[label]
                for label in coder_labels
            ]
        )

    return coded_matrix


def validate_labels(df):
    observed_labels = set(df[SENTIMENT_COLUMNS].stack().dropna())
    unexpected_labels = observed_labels - set(LABELS)

    if unexpected_labels:
        raise ValueError(f"Unexpected sentiment labels found: {sorted(unexpected_labels)}")


def add_metric(report_rows, metric, value, pair="", label_a="", label_b=""):
    report_rows.append(
        {
            "metric": metric,
            "pair": pair,
            "label_a": label_a,
            "label_b": label_b,
            "value": value,
        }
    )


def main():
    krippendorff = ensure_krippendorff()

    df = pd.read_csv(INPUT_FILE)
    df[SENTIMENT_COLUMNS] = df[SENTIMENT_COLUMNS].apply(lambda column: column.map(normalize_label))
    validate_labels(df)

    reliability_matrix = build_reliability_matrix(df)
    alpha_matrix = build_alpha_matrix(reliability_matrix)

    alpha = krippendorff.alpha(
        reliability_data=alpha_matrix,
        level_of_measurement="nominal",
    )

    report_rows = []
    add_metric(report_rows, "krippendorff_alpha_nominal", alpha)

    pairwise_kappas = []
    pairwise_matrices = {}

    for reviewer_a, reviewer_b in REVIEWER_PAIRS:
        pair_name = f"{reviewer_a} vs {reviewer_b}"
        pair_df = df[[reviewer_a, reviewer_b]].dropna()

        kappa = cohen_kappa_score(pair_df[reviewer_a], pair_df[reviewer_b], labels=LABELS)
        pairwise_kappas.append(kappa)
        add_metric(report_rows, "cohen_kappa", kappa, pair=pair_name)
        add_metric(report_rows, "pairwise_n_compared", len(pair_df), pair=pair_name)

        matrix = confusion_matrix(pair_df[reviewer_a], pair_df[reviewer_b], labels=LABELS)
        matrix_df = pd.DataFrame(matrix, index=LABELS, columns=LABELS)
        matrix_df.index.name = reviewer_a
        matrix_df.columns.name = reviewer_b
        pairwise_matrices[pair_name] = matrix_df

        for label_a in LABELS:
            for label_b in LABELS:
                add_metric(
                    report_rows,
                    "confusion_count",
                    int(matrix_df.loc[label_a, label_b]),
                    pair=pair_name,
                    label_a=label_a,
                    label_b=label_b,
                )

    average_kappa = sum(pairwise_kappas) / len(pairwise_kappas)
    add_metric(report_rows, "average_pairwise_cohen_kappa", average_kappa)

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(OUTPUT_FILE, index=False)

    print("Step 6 Reliability Summary")
    print(f"Rows analyzed: {len(df)}")
    print(f"Krippendorff's alpha (nominal): {alpha:.4f}")
    print(f"Average pairwise Cohen's kappa: {average_kappa:.4f}")

    print("\nPairwise Cohen's kappa:")
    for row in report_rows:
        if row["metric"] == "cohen_kappa":
            print(f"- {row['pair']}: {row['value']:.4f}")

    print("\nPairwise confusion matrices:")
    for pair_name, matrix_df in pairwise_matrices.items():
        print(f"\n{pair_name}")
        print(matrix_df)

    print(f"\nSaved reliability report to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
