import csv
import os
from collections import Counter

from tqdm import tqdm

from data_cleaning import extract_mda_from_html, split_into_sentences
from sec_connection import (
    download_filing_html,
    get_10k_filings,
    get_cik,
    get_filing_document_url,
)


TARGET_SENTENCES = 16500
MAX_SENTENCES_PER_COMPANY = 200
FILINGS_PER_COMPANY = 5
SLEEP_SECONDS = 0.15
DATA_COLLECTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(DATA_COLLECTION_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "sentences_raw.csv")

HEADERS = {
    "User-Agent": "Lucy Moore lmoore36@unc.edu",
    "Accept-Encoding": "gzip, deflate",
}

TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "INTC", "IBM", "ORCL", "CSCO", "ADBE",
    # Healthcare / Pharma
    "JNJ", "PFE", "MRK", "ABT", "BMY", "AMGN", "GILD", "MDT", "UNH", "CVS",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "COF", "USB",
    # Consumer / Retail
    "WMT", "AMZN", "TGT", "COST", "HD", "LOW", "NKE", "SBUX", "MCD", "YUM",
    # Energy
    "XOM", "CVX", "COP", "SLB", "PSX", "VLO", "MPC", "OXY", "HES", "DVN",
    # Industrials
    "GE", "HON", "MMM", "CAT", "DE", "BA", "LMT", "RTX", "UPS", "FDX",
    # Telecom / Media
    "T", "VZ", "CMCSA", "DIS", "NFLX", "PARA", "WBD", "FOXA", "DISH", "LUMN",
    # Materials / Misc
    "DD", "DOW", "LIN", "APD", "NEM", "FCX", "VMC", "MLM", "PKG", "IP",
]

CSV_COLUMNS = ["sentence_id", "sentence", "ticker", "cik", "filing_date", "filing_id"]


def load_existing_progress(output_file):
    """Return already-processed filing IDs and the next sentence number."""
    if not os.path.exists(output_file):
        return set(), 0, Counter()

    with open(output_file, "r", encoding="utf-8") as csvfile:
        rows = list(csv.DictReader(csvfile))

    existing_ids = {row.get("filing_id", "") for row in rows}
    company_counts = Counter(row.get("ticker", "") for row in rows)
    return existing_ids, len(rows), company_counts


def collect_sentences_for_filing(ticker, cik, filing):
    """Download one filing and return filtered MD&A sentences from it."""
    accession = filing["accession"]
    filing_date = filing["date"]

    print(f"  Processing {accession} ({filing_date})...")

    doc_url = get_filing_document_url(
        cik,
        accession,
        HEADERS,
        primary_document=filing.get("primary_document"),
        sleep_seconds=SLEEP_SECONDS,
    )
    if not doc_url:
        print("    Could not find document URL")
        return []

    print(f"    Document: {doc_url}")

    try:
        html = download_filing_html(doc_url, HEADERS, sleep_seconds=SLEEP_SECONDS)
    except Exception as e:
        print(f"    Could not download filing document: {e}")
        return []

    mda_text = extract_mda_from_html(html)
    if not mda_text:
        print("    Could not extract MD&A section")
        return []

    sentences = split_into_sentences(mda_text)
    print(f"    Extracted {len(sentences)} sentences")
    return sentences


def write_sentences(writer, sentences, sentence_counter, ticker, cik, filing):
    """Write extracted sentences to CSV and return the next sentence counter."""
    for sentence in sentences:
        writer.writerow({
            "sentence_id": f"s{sentence_counter:06d}",
            "sentence": sentence,
            "ticker": ticker,
            "cik": cik,
            "filing_date": filing["date"],
            "filing_id": filing["accession"],
        })
        sentence_counter += 1

    return sentence_counter


def main():
    print("=" * 60)
    print("SEC EDGAR 10-K MD&A Data Collection")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    existing_ids, sentence_counter, company_counts = load_existing_progress(OUTPUT_FILE)
    if existing_ids:
        print(f"Resuming - found {sentence_counter} sentences already collected.")

    mode = "a" if existing_ids else "w"
    with open(OUTPUT_FILE, mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        if not existing_ids:
            writer.writeheader()

        print(f"\nTarget: {TARGET_SENTENCES} sentences")
        print(f"Companies to process: {len(TICKERS)}")
        print(f"Filings per company: {FILINGS_PER_COMPANY}")
        print("-" * 60)

        for ticker in tqdm(TICKERS, desc="Companies"):
            if sentence_counter >= TARGET_SENTENCES:
                print(f"\nReached target of {TARGET_SENTENCES} sentences. Stopping.")
                break
            if company_counts[ticker] >= MAX_SENTENCES_PER_COMPANY:
                print(f"\n[{ticker}] Skipping - already has {company_counts[ticker]} sentences.")
                continue

            print(f"\n[{ticker}] Looking up CIK...")
            cik = get_cik(ticker, HEADERS, sleep_seconds=SLEEP_SECONDS)
            if not cik:
                print(f"  Skipping {ticker} - could not find CIK")
                continue

            print(f"  CIK: {cik}")
            filings = get_10k_filings(
                cik,
                HEADERS,
                max_filings=FILINGS_PER_COMPANY,
                sleep_seconds=SLEEP_SECONDS,
            )
            print(f"  Found {len(filings)} 10-K filings")

            for filing in filings:
                accession = filing["accession"]
                if accession in existing_ids:
                    print(f"  Skipping {accession} (already collected)")
                    continue

                sentences = collect_sentences_for_filing(ticker, cik, filing)
                remaining_for_company = MAX_SENTENCES_PER_COMPANY - company_counts[ticker]
                sentences = sentences[:remaining_for_company]
                sentence_counter = write_sentences(
                    writer,
                    sentences,
                    sentence_counter,
                    ticker,
                    cik,
                    filing,
                )
                company_counts[ticker] += len(sentences)
                csvfile.flush()
                existing_ids.add(accession)

                if sentence_counter >= TARGET_SENTENCES or company_counts[ticker] >= MAX_SENTENCES_PER_COMPANY:
                    break

            print(f"  Running total: {sentence_counter} sentences")

    print("\n" + "=" * 60)
    print(f"DONE. Collected {sentence_counter} sentences.")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
