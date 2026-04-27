import argparse
import asyncio
import logging
import os
import random
import re
import time
import warnings
from dataclasses import dataclass
from typing import Awaitable, Callable

import pandas as pd
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm


SYSTEM_PROMPT = """You are a research assistant classifying sentences from corporate 10-K annual report filings.

For each sentence, assign exactly one of these four labels:

EXPLORATION — the firm describes novelty, experimentation, new markets, new capabilities, R&D, pilots, or building something it does not yet possess. The outcome is uncertain. The capability or knowledge is genuinely new to the firm.

EXPLOITATION — the firm describes improving, scaling, optimizing, standardizing, or extracting more value from existing operations, products, or capabilities. The outcome is predictable. The capability already exists within the firm.

AMBIDEXTROUS — both EXPLORATION and EXPLOITATION are clearly and equally present in the same sentence. Both must be explicitly stated and structurally parallel. If one is secondary, assign the dominant label instead.

NEITHER — financial figures, legal boilerplate, risk disclosures, generic aspirational language, or any sentence with no clear strategic orientation.

Respond with ONLY the label. No explanation. No punctuation. Just one word: EXPLORATION, EXPLOITATION, AMBIDEXTROUS, or NEITHER."""

VALID_LABELS = {"EXPLORATION", "EXPLOITATION", "AMBIDEXTROUS", "NEITHER"}
OUTPUT_COLUMNS = ["sentence_id", "label", "model", "latency_seconds"]
SUMMARY_COLUMNS = [
    "model",
    "total_sentences",
    "total_time_seconds",
    "avg_latency_seconds",
    "estimated_cost_usd",
    "exploration_count",
    "exploitation_count",
    "ambidextrous_count",
    "neither_count",
]
REQUIRED_INPUT_COLUMNS = [
    "sentence_id",
    "sentence",
    "ticker",
    "industry",
    "filing_date",
    "filing_id",
]
BATCH_SIZE = 50
MAX_RETRIES = 8
MAX_OUTPUT_TOKENS = 256
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_project_root(start_dir):
    current_dir = start_dir
    while True:
        if os.path.isdir(os.path.join(current_dir, ".git")):
            return current_dir

        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            return start_dir
        current_dir = parent_dir


PROJECT_ROOT = find_project_root(SCRIPT_DIR)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str
    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float
    estimated_cost_per_1000_sentences: float = 0.0
    max_concurrency: int = 2
    min_request_interval_seconds: float = 0.5
    api_names: tuple[str, ...] = ()


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
    "claude-sonnet-4-6": ModelConfig("claude-sonnet-4-6", "anthropic", 0.003, 0.015, 1.50, 1, 1.4),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark AI model labels for the 10-K sentence holdout set."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to holdout_1000.csv. Defaults to common project holdout locations.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Folder for benchmark output. Per-model labels go in its data subfolder.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS),
        help="Run only one model, for example: python benchmark_labels.py --model gpt-4o",
    )
    return parser.parse_args()


def resolve_input_path(input_path):
    if input_path:
        if os.path.isabs(input_path) or os.path.exists(input_path):
            return input_path
        return os.path.join(PROJECT_ROOT, input_path)

    candidates = [
        "holdout_1000.csv",
        os.path.join("data_collection", "data", "holdout_1000.csv"),
        os.path.join("data-collection", "data", "holdout_1000.csv"),
        os.path.join("sec-data-collection", "data", "holdout_1000.csv"),
        os.path.join("sec_data_collection", "data", "holdout_1000.csv"),
    ]
    for candidate in candidates:
        cwd_candidate = os.path.abspath(candidate)
        project_candidate = os.path.join(PROJECT_ROOT, candidate)
        if os.path.exists(cwd_candidate):
            return cwd_candidate
        if os.path.exists(project_candidate):
            return project_candidate

    return os.path.join(PROJECT_ROOT, candidates[0])


def resolve_output_dir(output_dir):
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.join(PROJECT_ROOT, output_dir)


def labels_output_dir(output_dir):
    return os.path.join(output_dir, "data")


def load_holdout(input_path):
    df = pd.read_csv(input_path, dtype=str).fillna("")
    missing_columns = [column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Input file is missing required columns: {missing_columns}")
    return df


def normalize_label(raw_label, model_name, sentence_id):
    label = raw_label.strip().upper()
    if label in VALID_LABELS:
        return label

    logging.warning(
        "Invalid label from %s for sentence_id=%s: %r",
        model_name,
        sentence_id,
        raw_label,
    )
    return "INVALID"


def is_rate_limit_error(error):
    error_name = error.__class__.__name__.lower()
    error_message = str(error).lower()
    retry_terms = ["ratelimit", "rate limit", "429", "quota", "resource_exhausted"]
    return any(term in error_name or term in error_message for term in retry_terms)


def is_model_not_found_error(error):
    error_name = error.__class__.__name__.lower()
    error_message = str(error).lower()
    return "notfound" in error_name or "not found" in error_message or "404" in error_message


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


def estimate_tokens(text):
    # A rough fallback when provider usage metadata is unavailable.
    return max(1, len(text) // 4)


def sentence_prompt(sentence):
    return f"Classify this sentence:\n\n{sentence}"


async def call_anthropic(client, config, row):
    prompt = sentence_prompt(row["sentence"])
    start_time = time.perf_counter()
    response = await client.messages.create(
        model=config.name,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - start_time
    text = response.content[0].text if response.content else ""
    usage = response.usage
    return LabelResult(
        sentence_id=row["sentence_id"],
        label=normalize_label(text, config.name, row["sentence_id"]),
        model=config.name,
        latency_seconds=latency,
        input_tokens=getattr(usage, "input_tokens", 0),
        output_tokens=getattr(usage, "output_tokens", 0),
    )


async def call_openai(client, config, row):
    prompt = sentence_prompt(row["sentence"])
    start_time = time.perf_counter()
    response = await client.chat.completions.create(
        model=config.name,
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    latency = time.perf_counter() - start_time
    text = response.choices[0].message.content if response.choices else ""
    usage = response.usage
    return LabelResult(
        sentence_id=row["sentence_id"],
        label=normalize_label(text or "", config.name, row["sentence_id"]),
        model=config.name,
        latency_seconds=latency,
        input_tokens=getattr(usage, "prompt_tokens", 0),
        output_tokens=getattr(usage, "completion_tokens", 0),
    )


async def call_google(models, genai, config, row):
    prompt = f"{SYSTEM_PROMPT}\n\n{sentence_prompt(row['sentence'])}"
    response = None
    last_error = None
    start_time = time.perf_counter()

    for api_name, model in models:
        try:
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )
            break
        except Exception as error:
            if not is_model_not_found_error(error):
                raise
            last_error = error
            logging.warning(
                "Google model id %s was not available for %s; trying fallback.",
                api_name,
                config.name,
            )

    if response is None and last_error is not None:
        raise last_error
    if response is None:
        raise RuntimeError(f"No Google model ids configured for {config.name}.")

    latency = time.perf_counter() - start_time
    usage = getattr(response, "usage_metadata", None)
    try:
        text = response.text or ""
    except ValueError:
        text = ""

    return LabelResult(
        sentence_id=row["sentence_id"],
        label=normalize_label(text, config.name, row["sentence_id"]),
        model=config.name,
        latency_seconds=latency,
        input_tokens=getattr(usage, "prompt_token_count", 0),
        output_tokens=getattr(usage, "candidates_token_count", 0),
    )


def make_model_call(config, clients):
    if config.provider == "anthropic":
        client = clients["anthropic"]
        return lambda row: call_anthropic(client, config, row)
    if config.provider == "openai":
        client = clients["openai"]
        return lambda row: call_openai(client, config, row)
    if config.provider == "google":
        models = clients["google_models"][config.name]
        genai = clients["google_genai"]
        return lambda row: call_google(models, genai, config, row)
    raise ValueError(f"Unknown provider: {config.provider}")


def build_clients(configs):
    clients = {}
    providers = {config.provider for config in configs}

    if "anthropic" in providers:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic models.")
        clients["anthropic"] = AsyncAnthropic(api_key=api_key)

    if "openai" in providers:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI models.")
        clients["openai"] = AsyncOpenAI(api_key=api_key)

    if "google" in providers:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for Google models.")
        genai.configure(api_key=api_key)
        clients["google_genai"] = genai
        clients["google_models"] = {
            config.name: [(config.name, genai.GenerativeModel(config.name))]
            for config in configs
            if config.provider == "google"
        }

    return clients


def rows_to_records(rows):
    return rows.to_dict(orient="records")


def results_to_dataframe(results):
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
        columns=OUTPUT_COLUMNS,
    )


def summary_from_results_df(results_df, config, total_time=0.0):
    label_counts = results_df["label"].value_counts()
    input_tokens = sum(
        estimate_tokens(SYSTEM_PROMPT + sentence_prompt(sentence))
        for sentence in results_df.get("sentence", pd.Series(dtype=str)).astype(str)
    )
    output_tokens = sum(estimate_tokens(label) for label in results_df["label"].astype(str))
    token_estimated_cost = (
        (input_tokens / 1000) * config.input_cost_per_1k_tokens
        + (output_tokens / 1000) * config.output_cost_per_1k_tokens
    )
    run_estimated_cost = config.estimated_cost_per_1000_sentences * (len(results_df) / 1000)

    return {
        "model": config.name,
        "total_sentences": len(results_df),
        "total_time_seconds": round(total_time, 4),
        "avg_latency_seconds": round(results_df["latency_seconds"].mean(), 4),
        "estimated_cost_usd": round(max(token_estimated_cost, run_estimated_cost), 6),
        "exploration_count": int(label_counts.get("EXPLORATION", 0)),
        "exploitation_count": int(label_counts.get("EXPLOITATION", 0)),
        "ambidextrous_count": int(label_counts.get("AMBIDEXTROUS", 0)),
        "neither_count": int(label_counts.get("NEITHER", 0)),
    }


async def label_batch(rows, config, call_model):
    semaphore = asyncio.Semaphore(config.max_concurrency)
    pacer = RequestPacer(config.min_request_interval_seconds)

    async def label_one(row):
        async def paced_call():
            await pacer.wait()
            return await call_model(row)

        async with semaphore:
            return await retry_rate_limits(
                paced_call,
                model_name=config.name,
                sentence_id=row["sentence_id"],
            )

    tasks = [label_one(row) for row in rows_to_records(rows)]
    return await asyncio.gather(*tasks)


async def run_model(df, config, output_dir, clients):
    labels_dir = labels_output_dir(output_dir)
    os.makedirs(labels_dir, exist_ok=True)
    output_path = os.path.join(labels_dir, f"{config.name}_labels.csv")

    legacy_output_path = os.path.join(output_dir, f"{config.name}_labels.csv")
    if os.path.exists(legacy_output_path) and not os.path.exists(output_path):
        os.replace(legacy_output_path, output_path)

    existing_results_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    processed_sentence_ids = set()
    if os.path.exists(output_path):
        existing_results_df = pd.read_csv(output_path, dtype={"sentence_id": str})
        processed_sentence_ids = set(existing_results_df["sentence_id"].astype(str))
        if len(processed_sentence_ids) >= len(df):
            print(f"Skipping {config.name}: {output_path} already exists.")
            return summary_from_results_df(existing_results_df, config)
        print(
            f"Resuming {config.name}: {len(processed_sentence_ids)}/{len(df)} labels already saved."
        )

    call_model = make_model_call(config, clients)
    results = []
    input_tokens = 0
    output_tokens = 0
    start_time = time.perf_counter()
    remaining_df = df[~df["sentence_id"].astype(str).isin(processed_sentence_ids)]

    with tqdm(total=len(df), initial=len(processed_sentence_ids), desc=config.name, unit="sentence") as progress:
        for batch_start in range(0, len(remaining_df), BATCH_SIZE):
            batch = remaining_df.iloc[batch_start : batch_start + BATCH_SIZE]
            batch_results = await label_batch(batch, config, call_model)
            results.extend(batch_results)
            input_tokens += sum(result.input_tokens for result in batch_results)
            output_tokens += sum(result.output_tokens for result in batch_results)
            checkpoint_df = pd.concat(
                [existing_results_df, results_to_dataframe(results)],
                ignore_index=True,
            )
            checkpoint_df.to_csv(output_path, index=False)
            progress.update(len(batch_results))

    total_time = time.perf_counter() - start_time
    new_results_df = results_to_dataframe(results)
    results_df = pd.concat(
        [existing_results_df, new_results_df],
        ignore_index=True,
    )
    results_df.to_csv(output_path, index=False)

    if input_tokens == 0:
        input_tokens = sum(
            estimate_tokens(SYSTEM_PROMPT + sentence_prompt(sentence))
            for sentence in df["sentence"].astype(str)
        )
    if output_tokens == 0:
        output_tokens = sum(estimate_tokens(result.label) for result in results)

    token_estimated_cost = (
        (input_tokens / 1000) * config.input_cost_per_1k_tokens
        + (output_tokens / 1000) * config.output_cost_per_1k_tokens
    )
    run_estimated_cost = config.estimated_cost_per_1000_sentences * (len(results_df) / 1000)
    return summary_from_results_df(results_df, config, total_time)


def update_summary(output_dir, new_summary_rows):
    summary_path = os.path.join(output_dir, "benchmark_summary.csv")
    new_rows_df = pd.DataFrame(new_summary_rows, columns=SUMMARY_COLUMNS)

    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        if not new_rows_df.empty:
            summary_df = summary_df[~summary_df["model"].isin(new_rows_df["model"])]
            summary_df = pd.concat([summary_df, new_rows_df], ignore_index=True)
    else:
        summary_df = new_rows_df

    summary_df = summary_df.reindex(columns=SUMMARY_COLUMNS)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path}")


async def main_async():
    args = parse_args()
    load_dotenv()
    load_dotenv(os.path.join(SCRIPT_DIR, ".env"))
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

    input_path = resolve_input_path(args.input)
    output_dir = resolve_output_dir(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_dir, "benchmark_labels.log"),
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    df = load_holdout(input_path)
    selected_configs = [MODEL_CONFIGS[args.model]] if args.model else list(MODEL_CONFIGS.values())

    summary_rows = []
    for config in selected_configs:
        clients = build_clients([config])
        summary_row = await run_model(df, config, output_dir, clients)
        if summary_row:
            summary_rows.append(summary_row)

    update_summary(output_dir, summary_rows)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
