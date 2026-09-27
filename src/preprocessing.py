"""
src/preprocessing.py
Member 1: Data Preprocessing and Partitioning Module

Provides memory-efficient streaming and batch preprocessing for
massive TSV files with dynamic country partitioning.
"""

import os
from typing import Generator, Dict, Any, List
import pandas as pd
from src.normalization import normalize_record


def stream_normalized_records(
    file_path: str,
    chunk_size: int = 100_000,
    target_country: str = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream and yield normalized record dictionaries one by one from a TSV file.
    Optionally filter by target_country on the fly to save memory.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    for chunk in pd.read_csv(file_path, sep="\t", chunksize=chunk_size, dtype=str, keep_default_na=False):
        if target_country:
            chunk = chunk[chunk["country"].str.strip() == target_country]

        for record in chunk.to_dict(orient="records"):
            yield normalize_record(record)


def load_normalized_sample(
    file_path: str,
    n_rows: int = 10_000,
    target_country: str = None
) -> pd.DataFrame:
    """
    Convenience function to load a sample DataFrame with normalized columns.
    Useful for interactive experimentation and rapid testing.
    """
    records = []
    for rec in stream_normalized_records(file_path, chunk_size=min(n_rows, 50_000), target_country=target_country):
        records.append(rec)
        if len(records) >= n_rows:
            break

    return pd.DataFrame(records)
