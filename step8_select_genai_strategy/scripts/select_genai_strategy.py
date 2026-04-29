import argparse
import asyncio
import logging
import os
import random
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

# Keep Matplotlib/fontconfig caches inside the repo so local sandboxed runs work.
LOCAL_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR))

import krippendorff
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from tqdm import tqdm


SYSTEM_PROMPT = """You are a research assistant classifying sentences from corporate 10-K annual report filings.

For each sentence, assign exactly one of these four labels:

EXPLORATION — the firm describes novelty, experimentation, new markets, new capabilities, R&D, pilots, or building something it does not yet possess. The outcome is uncertain. The capability or knowledge is genuinely new to the firm.

EXPLOITATION — the firm describes improving, scaling, optimizing, standardizing, or extracting more value from existing operations, products, or capabilities. The outcome is predictable. The capability already exists within the firm.

AMBIDEXTROUS — both EXPLORATION and EXPLOITATION are clearly and equally present in the same sentence. Both must be explicitly stated and structurally parallel. If one is secondary, assign the dominant label instead.

NEITHER — financial figures, legal boilerplate, risk disclosures, generic aspirational language, or any sentence with no clear strategic orientation.

Respond with ONLY the label. No explanation. No punctuation. Just one word: EXPLORATION, EXPLOITATION, AMBIDEXTROUS, or NEITHER."""

HUMAN_TO_MODEL_LABEL = {
    "neither": "NEITHER",
    "exploitation": "EXPLOITATION",
    "exploration": "EXPLORATION",
    "ambiguous": "AMBIDEXTROUS",
    "ambidextrous": "AMBIDEXTROUS",
}
MODEL_LABELS = ["NEITHER", "EXPLOITATION", "EXPLORATION", "AMBIDEXTROUS"]
LABEL_TO_CODE = {label: index for index, label in enumerate(MODEL_LABELS)}
VALID_LABELS = set(MODEL_LABELS)
LABEL_OUTPUT_COLUMNS = ["sentence_id", "label", "model", "latency_seconds"]
BATCH_SIZE = 50
MAX_RETRIES = 8
MAX_OUTPUT_TOKENS = 256


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str
    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float
    estimated_cost_per_1000_sentences: float = 0.0
    max_concurrency: int = 2
    min_request_interval_seconds: float = 0.5


@dataclass
class LabelResult:
    sentence_id: str
    label: str
    model: str
    latency_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0


MODEL_CONFIGS = {
    "gpt-4o-mini": ModelConfig("gpt-4o-mini", "openai", 0.00015, 0.0006, 0.05, 5, 0.2),
    "gpt-4o": ModelConfig("gpt-4o", "openai", 0.005, 0.015, 2.50, 1, 1.2),
    "gemini-2.5-flash": ModelConfig("gemini-2.5-flash", "google", 0.00015, 0.00060),
    "gemini-2.5-flash-lite": ModelConfig(
        "gemini-2.5-flash-lite", "google", 0.000075, 0.0003
    ),
    "claude-haiku-4-5-20251001": ModelConfig(
        "claude-haiku-4-5-20251001", "anthropic", 0.00025, 0.00125, 0.15, 1, 1.4
    ),
    "claude-sonnet-4-6": ModelConfig(
        "claude-sonnet-4-6", "anthropic", 0.003, 0.015, 1.50, 1, 1.4
    ),
}


def find_project_root(start_dir):
    current_dir = Path(start_dir).resolve()
    while True:
        if (current_dir / ".git").is_dir():
            return current_dir
        if current_dir.parent == current_dir:
            return Path(start_dir).resolve()
        current_dir = current_dir.parent


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)
DEFAULT_STEP8_DIR = PROJECT_ROOT / "step8_select_genai_strategy"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare GenAI benchmark labels, select a strategy, and optionally label train/test."
    )
    parser.add_argument(
        "--human-consensus",
        default=PROJECT_ROOT / "step6_human_labeling" / "data" / "holdout_with_consensus.csv",
        type=Path,
        help="Path to Step 6 holdout_with_consensus.csv.",
    )
    parser.add_argument(
        "--labels-dir",
        default=PROJECT_ROOT / "step7_genai_benchmark" / "data",
        type=Path,
        help="Directory containing Step 7 *_labels.csv files.",
    )
    parser.add_argument(
        "--benchmark-summary",
        default=PROJECT_ROOT / "step7_genai_benchmark" / "benchmark_summary.csv",
        type=Path,
        help="Path to Step 7 benchmark_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_STEP8_DIR / "data",
        type=Path,
        help="Directory for Step 8 outputs.",
    )
    parser.add_argument(
        "--selection-metric",
        default="macro_f1",
        choices=["macro_f1", "krippendorff_alpha", "mcc", "accuracy", "weighted_f1"],
        help="Metric used to select the labeling model.",
    )
    parser.add_argument(
        "--label-train-test",
        action="store_true",
        help="Also label Step 5 train/test files with the selected model. This uses paid APIs.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS),
        help="Override automatic model selection for train/test labeling.",
    )
    parser.add_argument(
        "--train-file",
        default=PROJECT_ROOT / "step5_preprocess_split" / "data" / "train.csv",
        type=Path,
        help="Path to Step 5 train.csv.",
    )
    parser.add_argument(
        "--test-file",
        default=PROJECT_ROOT / "step5_preprocess_split" / "data" / "test.csv",
        type=Path,
        help="Path to Step 5 test.csv.",
    )
    parser.add_argument(
        "--train-test-file",
        default=PROJECT_ROOT / "step5_preprocess_split" / "data" / "train_test_15000.csv",
        type=Path,
        help="Path to Step 5 train_test_15000.csv.",
    )
    return parser.parse_args()


def require_file(path, description):
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def normalize_human_label(label):
    normalized = str(label).strip().lower()
    return HUMAN_TO_MODEL_LABEL.get(normalized, "INVALID")


def load_human_consensus(path):
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    required = {"sentence_id", "consensus"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df["human_label"] = df["consensus"].map(normalize_human_label)
    invalid = sorted(set(df["human_label"]) - set(MODEL_LABELS))
    if invalid:
        raise ValueError(f"Unexpected human labels after normalization: {invalid}")

    return df[["sentence_id", "human_label"]]


def load_model_labels(label_file):
    df = pd.read_csv(label_file, dtype=str, encoding="utf-8-sig").fillna("")
    required = {"sentence_id", "label", "model"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label_file} is missing required columns: {missing}")

    df["label"] = df["label"].str.strip().str.upper()
    return df


def krippendorff_alpha_two_raters(human_labels, model_labels):
    coded = [
        [LABEL_TO_CODE[label] for label in human_labels],
        [LABEL_TO_CODE[label] for label in model_labels],
    ]
    return krippendorff.alpha(reliability_data=coded, level_of_measurement="nominal")


def save_confusion_matrix(output_dir, model_name, y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=MODEL_LABELS)
    matrix_df = pd.DataFrame(matrix, index=MODEL_LABELS, columns=MODEL_LABELS)

    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} vs Human Consensus")
    plt.ylabel("Human consensus")
    plt.xlabel("Model label")
    plt.tight_layout()

    safe_model = model_name.replace(".", "_").replace("/", "_")
    output_path = output_dir / f"confusion_matrix_{safe_model}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def evaluate_model(label_file, human_df, output_dir):
    model_df = load_model_labels(label_file)
    model_name = model_df["model"].replace("", pd.NA).dropna().iloc[0]
    merged = human_df.merge(model_df, on="sentence_id", how="inner")
    merged = merged[merged["label"].isin(MODEL_LABELS)].copy()

    if merged.empty:
        raise ValueError(f"No comparable labels found for {model_name} in {label_file}")

    y_true = merged["human_label"]
    y_pred = merged["label"]
    report = classification_report(
        y_true,
        y_pred,
        labels=MODEL_LABELS,
        output_dict=True,
        zero_division=0,
    )
    alpha = krippendorff_alpha_two_raters(y_true.tolist(), y_pred.tolist())
    confusion_path = save_confusion_matrix(output_dir, model_name, y_true, y_pred)

    return {
        "model": model_name,
        "n_compared": len(merged),
        "accuracy": accuracy_score(y_true, y_pred),
        "krippendorff_alpha": alpha,
        "cohen_kappa": cohen_kappa_score(y_true, y_pred, labels=MODEL_LABELS),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=MODEL_LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            y_true, y_pred, labels=MODEL_LABELS, average="weighted", zero_division=0
        ),
        "f1_neither": report["NEITHER"]["f1-score"],
        "f1_exploitation": report["EXPLOITATION"]["f1-score"],
        "f1_exploration": report["EXPLORATION"]["f1-score"],
        "f1_ambiguous": report["AMBIDEXTROUS"]["f1-score"],
        "confusion_matrix_file": str(confusion_path),
    }


def build_benchmark_results(args):
    require_file(args.human_consensus, "Human consensus file")
    require_file(args.benchmark_summary, "Benchmark summary file")
    if not args.labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {args.labels_dir}")

    label_files = sorted(args.labels_dir.glob("*_labels.csv"))
    if not label_files:
        raise FileNotFoundError(f"No *_labels.csv files found in {args.labels_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    human_df = load_human_consensus(args.human_consensus)
    summary_df = pd.read_csv(args.benchmark_summary, encoding="utf-8-sig")

    rows = [evaluate_model(label_file, human_df, args.output_dir) for label_file in label_files]
    results_df = pd.DataFrame(rows)
    results_df = results_df.merge(summary_df, on="model", how="left")
    results_df["alpha_per_dollar"] = results_df.apply(
        lambda row: row["krippendorff_alpha"] / row["estimated_cost_usd"]
        if pd.notna(row.get("estimated_cost_usd")) and row["estimated_cost_usd"] > 0
        else pd.NA,
        axis=1,
    )
    results_df = results_df.sort_values(
        [args.selection_metric, "krippendorff_alpha", "mcc"],
        ascending=False,
    )

    output_file = args.output_dir / "step8_benchmark_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Saved benchmark results to: {output_file}")
    return results_df


def write_selected_strategy(args, benchmark_results):
    selected_row = benchmark_results.sort_values(
        [args.selection_metric, "krippendorff_alpha", "mcc"],
        ascending=False,
    ).iloc[0]
    selected_model = args.model or selected_row["model"]

    strategy_text = f"""Selected GenAI labeling strategy
================================
Selected model: {selected_model}
Selection metric: {args.selection_metric}
{args.selection_metric}: {selected_row[args.selection_metric]}
Krippendorff alpha: {selected_row['krippendorff_alpha']}
Cohen kappa: {selected_row['cohen_kappa']}
MCC: {selected_row['mcc']}
Macro F1: {selected_row['macro_f1']}
Estimated cost USD on holdout: {selected_row.get('estimated_cost_usd', 'n/a')}
"""
    output_file = args.output_dir / "selected_labeling_strategy.txt"
    output_file.write_text(strategy_text, encoding="utf-8")
    print(strategy_text)
    print(f"Saved selected strategy to: {output_file}")
    return selected_model


def normalize_model_label(raw_label, model_name, sentence_id):
    label = raw_label.strip().upper()
    if label in VALID_LABELS:
        return label

    logging.warning("Invalid label from %s for sentence_id=%s: %r", model_name, sentence_id, raw_label)
    return "INVALID"


def sentence_prompt(sentence):
    return f"Classify this sentence:\n\n{sentence}"


def is_rate_limit_error(error):
    error_name = error.__class__.__name__.lower()
    error_message = str(error).lower()
    retry_terms = ["ratelimit", "rate limit", "429", "quota", "resource_exhausted"]
    return any(term in error_name or term in error_message for term in retry_terms)


def retry_delay_seconds(error, attempt):
    error_message = str(error).lower()
    milliseconds_match = re.search(r"try again in ([0-9.]+)ms", error_message)
    if milliseconds_match:
        return max(2.0, float(milliseconds_match.group(1)) / 1000)

    seconds_match = re.search(r"try again in ([0-9.]+)s", error_message)
    if seconds_match:
        return max(2.0, float(seconds_match.group(1)))

    return min(60.0, (2**attempt) + random.uniform(0, 0.5))


class RequestPacer:
    def __init__(self, min_interval_seconds):
        self.min_interval_seconds = min_interval_seconds
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        if self.min_interval_seconds <= 0:
            return

        async with self.lock:
            elapsed = time.perf_counter() - self.last_request_time
            if elapsed < self.min_interval_seconds:
                await asyncio.sleep(self.min_interval_seconds - elapsed)
            self.last_request_time = time.perf_counter()


async def retry_rate_limits(call: Callable[[], Awaitable[LabelResult]], model_name, sentence_id):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await call()
        except Exception as error:
            if not is_rate_limit_error(error) or attempt == MAX_RETRIES:
                raise
            sleep_seconds = retry_delay_seconds(error, attempt)
            logging.warning(
                "Rate limit for %s sentence_id=%s. Retry %s/%s in %.2fs.",
                model_name,
                sentence_id,
                attempt + 1,
                MAX_RETRIES,
                sleep_seconds,
            )
            await asyncio.sleep(sleep_seconds)


async def call_openai(client, config, row):
    start_time = time.perf_counter()
    response = await client.chat.completions.create(
        model=config.name,
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sentence_prompt(row["sentence"])},
        ],
    )
    latency = time.perf_counter() - start_time
    text = response.choices[0].message.content if response.choices else ""
    usage = response.usage
    return LabelResult(
        row["sentence_id"],
        normalize_model_label(text or "", config.name, row["sentence_id"]),
        config.name,
        latency,
        getattr(usage, "prompt_tokens", 0),
        getattr(usage, "completion_tokens", 0),
    )


async def call_anthropic(client, config, row):
    start_time = time.perf_counter()
    response = await client.messages.create(
        model=config.name,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": sentence_prompt(row["sentence"])}],
    )
    latency = time.perf_counter() - start_time
    text = response.content[0].text if response.content else ""
    usage = response.usage
    return LabelResult(
        row["sentence_id"],
        normalize_model_label(text, config.name, row["sentence_id"]),
        config.name,
        latency,
        getattr(usage, "input_tokens", 0),
        getattr(usage, "output_tokens", 0),
    )


async def call_google(model, genai, config, row):
    prompt = f"{SYSTEM_PROMPT}\n\n{sentence_prompt(row['sentence'])}"
    start_time = time.perf_counter()
    response = await model.generate_content_async(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )
    latency = time.perf_counter() - start_time
    usage = getattr(response, "usage_metadata", None)
    try:
        text = response.text or ""
    except ValueError:
        text = ""
    return LabelResult(
        row["sentence_id"],
        normalize_model_label(text, config.name, row["sentence_id"]),
        config.name,
        latency,
        getattr(usage, "prompt_token_count", 0),
        getattr(usage, "candidates_token_count", 0),
    )


def build_model_client(config):
    if config.provider == "openai":
        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the selected OpenAI model.")
        client = AsyncOpenAI(api_key=api_key)
        return lambda row: call_openai(client, config, row)

    if config.provider == "anthropic":
        from anthropic import AsyncAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the selected Anthropic model.")
        client = AsyncAnthropic(api_key=api_key)
        return lambda row: call_anthropic(client, config, row)

    if config.provider == "google":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for the selected Google model.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(config.name)
        return lambda row: call_google(model, genai, config, row)

    raise ValueError(f"Unknown provider: {config.provider}")


def label_results_to_dataframe(results):
    return pd.DataFrame(
        [
            {
                "sentence_id": result.sentence_id,
                "label": result.label,
                "model": result.model,
                "latency_seconds": round(result.latency_seconds, 4),
            }
            for result in results
        ],
        columns=LABEL_OUTPUT_COLUMNS,
    )


async def label_dataset(input_file, labels_file, output_file, config):
    source_df = pd.read_csv(input_file, dtype=str, encoding="utf-8-sig").fillna("")
    existing_labels = pd.DataFrame(columns=LABEL_OUTPUT_COLUMNS)
    processed_ids = set()

    if labels_file.exists():
        existing_labels = pd.read_csv(labels_file, dtype={"sentence_id": str}, encoding="utf-8-sig")
        processed_ids = set(existing_labels["sentence_id"].astype(str))
        print(f"Resuming {labels_file.name}: {len(processed_ids)}/{len(source_df)} labels already saved.")

    remaining_df = source_df[~source_df["sentence_id"].astype(str).isin(processed_ids)]
    call_model = build_model_client(config)
    semaphore = asyncio.Semaphore(config.max_concurrency)
    pacer = RequestPacer(config.min_request_interval_seconds)
    new_results = []

    async def label_one(row):
        async def paced_call():
            await pacer.wait()
            return await call_model(row)

        async with semaphore:
            return await retry_rate_limits(paced_call, config.name, row["sentence_id"])

    with tqdm(total=len(source_df), initial=len(processed_ids), desc=labels_file.name, unit="sentence") as progress:
        for batch_start in range(0, len(remaining_df), BATCH_SIZE):
            batch = remaining_df.iloc[batch_start : batch_start + BATCH_SIZE]
            batch_results = await asyncio.gather(
                *[label_one(row) for row in batch.to_dict(orient="records")]
            )
            new_results.extend(batch_results)
            checkpoint = pd.concat(
                [existing_labels, label_results_to_dataframe(new_results)],
                ignore_index=True,
            )
            checkpoint.to_csv(labels_file, index=False)
            progress.update(len(batch_results))

    final_labels = pd.concat(
        [existing_labels, label_results_to_dataframe(new_results)],
        ignore_index=True,
    )
    final_labels.to_csv(labels_file, index=False)

    labeled_df = source_df.merge(
        final_labels[["sentence_id", "label", "model"]],
        on="sentence_id",
        how="left",
    )
    labeled_df = labeled_df.rename(columns={"label": "genai_label", "model": "genai_model"})
    labeled_df.to_csv(output_file, index=False)
    return labeled_df


async def label_train_test(args, selected_model):
    for path, description in [
        (args.train_file, "Train file"),
        (args.test_file, "Test file"),
        (args.train_test_file, "Train/test file"),
    ]:
        require_file(path, description)

    config = MODEL_CONFIGS[selected_model]
    safe_model = selected_model.replace(".", "_").replace("/", "_")
    jobs = [
        (
            args.train_file,
            args.output_dir / f"train_labeled_{safe_model}_labels.csv",
            args.output_dir / f"train_labeled_{safe_model}.csv",
        ),
        (
            args.test_file,
            args.output_dir / f"test_labeled_{safe_model}_labels.csv",
            args.output_dir / f"test_labeled_{safe_model}.csv",
        ),
        (
            args.train_test_file,
            args.output_dir / f"train_test_labeled_{safe_model}_labels.csv",
            args.output_dir / f"train_test_labeled_{safe_model}.csv",
        ),
    ]

    for input_file, labels_file, output_file in jobs:
        await label_dataset(input_file, labels_file, output_file, config)
        print(f"Saved labeled dataset to: {output_file}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv(DEFAULT_STEP8_DIR / ".env")
    load_dotenv(PROJECT_ROOT / ".env")
    logging.basicConfig(
        filename=DEFAULT_STEP8_DIR / "step8_label_train_test.log",
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    benchmark_results = build_benchmark_results(args)
    selected_model = write_selected_strategy(args, benchmark_results)

    if args.label_train_test:
        asyncio.run(label_train_test(args, selected_model))
    else:
        print("Train/test labeling skipped. Pass --label-train-test to run paid API labeling.")


if __name__ == "__main__":
    main()
