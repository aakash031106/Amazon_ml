"""
src/normalization.py
Member 1: Normalization & Preprocessing Module

Provides clean, standardized versions of business names and addresses
without discarding original text.
"""

import re
import unicodedata
from typing import Dict, Any, List

# 1. Common Legal Suffixes dictionary
LEGAL_SUFFIXES = {
    r"\bprivate\s+limited\b": "pvt ltd",
    r"\bpvt\s*\.?\s*ltd\s*\.?\b": "pvt ltd",
    r"\bincorporated\b": "inc",
    r"\binc\s*\.?\b": "inc",
    r"\bcorporation\b": "corp",
    r"\bcorp\s*\.?\b": "corp",
    r"\blimited\b": "ltd",
    r"\bltd\s*\.?\b": "ltd",
    r"\blimited\s+liability\s+company\b": "llc",
    r"\bllc\s*\.?\b": "llc",
    r"\bllp\s*\.?\b": "llp",
    r"\bcompany\b": "co",
    r"\bco\s*\.?\b": "co",
    r"\bsociety\s+anonyme\b": "sa",
    r"\bsarl\b": "sarl",
    r"\bsas\b": "sas",
}

# 2. Common Address Abbreviations
ADDRESS_ABBREVIATIONS = {
    r"\bstreet\b": "st",
    r"\bavenue\b": "ave",
    r"\broad\b": "rd",
    r"\bdrive\b": "dr",
    r"\blane\b": "ln",
    r"\bboulevard\b": "blvd",
    r"\bhighway\b": "hwy",
    r"\bsuite\b": "ste",
    r"\bapartment\b": "apt",
    r"\bbuilding\b": "bldg",
    r"\bfloor\b": "fl",
    r"\bopposite\b": "opp",
    r"\bnear\b": "nr",
}


def strip_accents_and_lower(text: str) -> str:
    """
    Convert text to lowercase and decompose accents/diacritics.
    Example: 'Payne Énterprises' -> 'payne enterprises'
    """
    if not text:
        return ""
    # Unicode NFKD decomposes accented characters like É into E + accent
    nfkd_form = unicodedata.normalize("NFKD", str(text))
    # Keep only characters that are not accent combining marks
    text_no_accents = "".join(c for c in nfkd_form if not unicodedata.combining(c))
    return text_no_accents.lower().strip()


def clean_artifacts_and_punct(text: str) -> str:
    """
    Remove literal 'null' strings, replace punctuation with spaces,
    and collapse multiple spaces.
    Example: 'peoria, il, null, 630' -> 'peoria il 630'
    """
    if not text:
        return ""
    # Remove literal "null" or "none" surrounded by non-alphanumeric
    text = re.sub(r"\b(null|none)\b", " ", text, flags=re.IGNORECASE)
    # Replace punctuation with single space (except letters, digits, and basic Indic Unicode)
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespaces to single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_business_name(name: str) -> str:
    """
    Full normalization pipeline for business names:
    1. Lowercase + accent removal
    2. Standardize legal suffixes (incorporated -> inc, etc.)
    3. Clean punctuation and spaces
    """
    if not name:
        return ""
    text = strip_accents_and_lower(name)
    for pattern, replacement in LEGAL_SUFFIXES.items():
        text = re.sub(pattern, replacement, text)
    text = clean_artifacts_and_punct(text)
    return text


def normalize_address(address: str) -> str:
    """
    Full normalization pipeline for addresses:
    1. Lowercase + accent removal
    2. Standardize address terms (street -> st, road -> rd, etc.)
    3. Clean punctuation and spaces
    """
    if not address:
        return ""
    text = strip_accents_and_lower(address)
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)
    text = clean_artifacts_and_punct(text)
    return text


def extract_numbers(text: str) -> List[str]:
    """
    Extract all numeric tokens (house numbers, PIN codes, zip codes).
    Example: '3315 Fremont St Apt 4B' -> ['3315', '4']
    """
    if not text:
        return []
    return re.findall(r"\b\d+\b", text)


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take a raw record dictionary with:
      - entity_id
      - business_name
      - business_address
      - country
    Return an enriched dictionary preserving original fields and adding:
      - norm_name
      - norm_address
      - name_tokens
      - address_tokens
      - address_numbers
      - is_address_empty
    """
    raw_name = record.get("business_name", "") or ""
    raw_addr = record.get("business_address", "") or ""
    country = (record.get("country", "") or "").strip()

    norm_name = normalize_business_name(raw_name)
    norm_addr = normalize_address(raw_addr)

    return {
        "entity_id": record.get("entity_id", ""),
        "business_name": raw_name,
        "business_address": raw_addr,
        "country": country,
        "norm_name": norm_name,
        "norm_address": norm_addr,
        "name_tokens": norm_name.split() if norm_name else [],
        "address_tokens": norm_addr.split() if norm_addr else [],
        "address_numbers": extract_numbers(norm_addr),
        "is_address_empty": (raw_addr.strip() == ""),
    }
