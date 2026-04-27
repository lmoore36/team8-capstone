import html as html_module
import re
import unicodedata
import warnings

from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize


warnings.filterwarnings("ignore")


def ensure_sentence_tokenizer():
    """Download NLTK sentence tokenizer data if it is not already present."""
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def extract_mda_from_html(html):
    """Extract the MD&A / Item 7 section from one 10-K HTML document."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(["table", "script", "style", "ix:header"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = html_module.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\s+", " ", text)

    start_patterns = [
        r"item\s*7[\.\:\-\u2013]?\s*management'?s?\s+discussion\s+and\s+analysis\s+of\s+financial\s+condition\s+and\s+results\s+of\s+operations",
        r"item\s*7[\.\:\-\u2013]?\s*management'?s?\s+discussion\s+and\s+analysis",
        r"management'?s?\s+discussion\s+and\s+analysis",
    ]
    end_patterns = [
        r"item\s*7a[\.\:\-\u2013]?\s*quantitative\s+and\s+qualitative\s+disclosures",
        r"item\s*8[\.\:\-\u2013]?\s*financial\s+statements",
    ]

    best_section = ""
    for start_pattern in start_patterns:
        for start_match in re.finditer(start_pattern, text, re.IGNORECASE):
            start_pos = start_match.start()
            search_from = start_match.end() + 100
            end_pos = min(start_pos + 150000, len(text))

            for end_pattern in end_patterns:
                end_match = re.search(end_pattern, text[search_from:], re.IGNORECASE)
                if end_match:
                    candidate_end = search_from + end_match.start()
                    if candidate_end < end_pos:
                        end_pos = candidate_end

            candidate = text[start_pos:end_pos].strip()

            # Table-of-contents hits usually run only a few words before Item 7A/8.
            if len(candidate) > len(best_section):
                best_section = candidate
            if len(candidate) > 3000:
                return candidate

    return best_section if len(best_section) > 500 else None


def split_into_sentences(text):
    """Split MD&A text into clean sentence-level examples for labeling."""
    ensure_sentence_tokenizer()

    sentences = sent_tokenize(text)
    clean = []

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 40:
            continue
        if len(sentence) > 600:
            continue

        letters = sum(char.isalpha() for char in sentence)
        if letters / max(len(sentence), 1) < 0.45:
            continue

        skip_phrases = [
            "item 7",
            "item 8",
            "management's discussion",
            "forward-looking statements",
            "table of contents",
            "see notes to consolidated",
            "incorporated by reference",
        ]
        if any(phrase in sentence.lower() for phrase in skip_phrases):
            continue

        clean.append(sentence)

    return clean
