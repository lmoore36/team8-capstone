import re
import time
from urllib.parse import parse_qs, unquote, urljoin, urlsplit

import requests


def request_sec(url, headers, params=None, timeout=10, sleep_seconds=0.15):
    """Make one SEC request and pause so we stay inside EDGAR rate guidance."""
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    time.sleep(sleep_seconds)
    response.raise_for_status()
    return response


def clean_doc_url(url):
    """Strip the iXBRL viewer wrapper SEC adds to some filing document URLs."""
    if not url:
        return url

    parsed = urlsplit(url)
    if parsed.path == "/ix":
        doc = parse_qs(parsed.query).get("doc", [""])[0]
        if doc:
            return urljoin("https://www.sec.gov", unquote(doc))

    return url


def get_cik(ticker, headers, sleep_seconds=0.15):
    """Look up EDGAR's internal company ID for a ticker symbol."""
    cik = get_cik_from_ticker_file(ticker, headers, sleep_seconds=sleep_seconds)
    if cik:
        return cik

    return get_cik_from_company_search(ticker, headers, sleep_seconds=sleep_seconds)


def get_cik_from_ticker_file(ticker, headers, sleep_seconds=0.15):
    """Use SEC's official ticker/CIK/company-name JSON mapping."""
    url = "https://www.sec.gov/files/company_tickers.json"

    try:
        response = request_sec(url, headers, sleep_seconds=sleep_seconds)
        for company in response.json().values():
            if company.get("ticker", "").upper() == ticker.upper():
                return str(company["cik_str"]).zfill(10)
    except Exception as e:
        print(f"  Could not get CIK from ticker file for {ticker}: {e}")

    return None


def get_cik_from_company_search(ticker, headers, sleep_seconds=0.15):
    """Fallback CIK lookup using the older company search page."""
    url = "https://www.sec.gov/cgi-bin/browse-edgar"
    params = {
        "CIK": ticker,
        "type": "10-K",
        "action": "getcompany",
        "output": "atom",
    }

    try:
        response = request_sec(url, headers, params=params, sleep_seconds=sleep_seconds)
        match = re.search(r"CIK=(\d+)", response.url + response.text)
        if match:
            return match.group(1).zfill(10)
    except Exception as e:
        print(f"  Could not get CIK for {ticker}: {e}")

    return None


def get_10k_filings(cik, headers, max_filings=5, sleep_seconds=0.15):
    """Return recent 10-K filing accession numbers and dates for one CIK."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    try:
        response = request_sec(url, headers, sleep_seconds=sleep_seconds)
        data = response.json()
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        dates = filings.get("filingDate", [])
        primary_documents = filings.get("primaryDocument", [])

        results = []
        for form, accession, date, primary_document in zip(
            forms,
            accessions,
            dates,
            primary_documents,
        ):
            if form == "10-K":
                results.append({
                    "accession": accession,
                    "date": date,
                    "primary_document": primary_document,
                })
            if len(results) >= max_filings:
                break

        return results
    except Exception as e:
        print(f"  Could not get filings for CIK {cik}: {e}")
        return []


def get_filing_document_url(
    cik,
    accession_number,
    headers,
    primary_document=None,
    sleep_seconds=0.15,
):
    """Build the raw 10-K document URL from SEC's primaryDocument metadata."""
    del headers, sleep_seconds

    if not primary_document:
        print(f"  Missing primary document for {accession_number}")
        return None

    acc_clean = accession_number.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/"
    return clean_doc_url(urljoin(base, primary_document))


def download_filing_html(doc_url, headers, sleep_seconds=0.15):
    """Download the raw filing HTML after normalizing any iXBRL viewer URL."""
    response = request_sec(
        clean_doc_url(doc_url),
        headers,
        timeout=30,
        sleep_seconds=sleep_seconds,
    )
    return response.text
