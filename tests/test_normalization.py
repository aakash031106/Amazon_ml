"""
tests/test_normalization.py
Unit tests for Member 1's Normalization module.
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.normalization import (
    strip_accents_and_lower,
    clean_artifacts_and_punct,
    normalize_business_name,
    normalize_address,
    extract_numbers,
    normalize_record
)

def test_normalization():
    print("Running Normalization Tests...\n")

    # 1. Accent Stripping
    sample_accent = "Payne Énterprises"
    clean_accent = strip_accents_and_lower(sample_accent)
    print(f"Accent Test: '{sample_accent}' -> '{clean_accent}'")
    assert clean_accent == "payne enterprises", f"Failed: got {clean_accent}"

    # 2. Legal Suffixes
    sample_legal = "Acme Incorporated"
    clean_legal = normalize_business_name(sample_legal)
    print(f"Legal Suffix Test 1: '{sample_legal}' -> '{clean_legal}'")
    assert clean_legal == "acme inc", f"Failed: got {clean_legal}"

    sample_pvt = "Ss Food Private Limited"
    clean_pvt = normalize_business_name(sample_pvt)
    print(f"Legal Suffix Test 2: '{sample_pvt}' -> '{clean_pvt}'")
    assert clean_pvt == "ss food pvt ltd", f"Failed: got {clean_pvt}"

    # 3. Address Normalization & Artifact Removal
    sample_addr = "630 45th Terrace, Kansas City, MO, null"
    clean_addr = normalize_address(sample_addr)
    print(f"Address Artifact Test: '{sample_addr}' -> '{clean_addr}'")
    assert "null" not in clean_addr, "Failed: 'null' artifact remained!"
    assert clean_addr == "630 45th terrace kansas city mo", f"Failed: got {clean_addr}"

    # 4. Street Abbreviation
    sample_street = "3315 Fremont Street, Peoria, IL"
    clean_street = normalize_address(sample_street)
    print(f"Street Abbr Test: '{sample_street}' -> '{clean_street}'")
    assert clean_street == "3315 fremont st peoria il", f"Failed: got {clean_street}"

    # 5. Extract Numbers
    sample_num = "No. 75, Natarajan St, Vadapalani, PIN 600026"
    nums = extract_numbers(sample_num)
    print(f"Number Extraction Test: '{sample_num}' -> {nums}")
    assert "75" in nums and "600026" in nums, f"Failed: got {nums}"

    # 6. Full Record Normalization
    raw_rec = {
        "entity_id": "S1-12345",
        "business_name": "Dréxkor International Ltd.",
        "business_address": "85 Wayne Avenue, Ticonderoga, NY",
        "country": "US"
    }
    enriched = normalize_record(raw_rec)
    print(f"\nFull Record Normalization Result:")
    for k, v in enriched.items():
        print(f"  {k}: {v}")

    assert enriched["norm_name"] == "drexkor international ltd"
    assert enriched["norm_address"] == "85 wayne ave ticonderoga ny"
    assert "85" in enriched["address_numbers"]
    assert enriched["is_address_empty"] is False

    print("\nALL NORMALIZATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_normalization()
