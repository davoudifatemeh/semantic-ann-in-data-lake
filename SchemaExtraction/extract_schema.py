import pandas as pd
import json
import os
import glob
from typing import Dict, Any
import re
import requests
from collections import defaultdict
import itertools

def infer_dtype(series: pd.Series) -> str:
    # Map pandas dtype to schema-friendly dtype
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    elif pd.api.types.is_float_dtype(series):
        return "float"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    else:
        return "string"

def get_target_tables(ground_truth_csv) -> set:
    ground_truth = pd.read_csv(ground_truth_csv)
    candidate_tables = ground_truth["candidate_table"].unique().astype(str).tolist()
    query_tables = ground_truth["query_table"].unique().astype(str).tolist()

    tables = set(candidate_tables) | set(query_tables)

    print(f"Loaded {len(tables)} target tables from {ground_truth_csv}")
    return tables


def summarize_csv(file_path, distinct_sample_size=30, row_sample_size=3) -> Dict[str, Any]:
    # Summarize a single CSV file into schema + sample values + sample rows
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None  # skip unreadable files

    summary = {
        "file": os.path.basename(file_path),
        "row_count": len(df),
        "columns": [],
        "values_sample": {},
        "sample_rows": df.head(row_sample_size).to_dict(orient="records")
    }

    for col in df.columns:
        col_data = df[col].dropna().astype(str)
        num_values = len(col_data)
        lengths = col_data.str.len()
        # Basic column stats
        if not lengths.empty:
            max_len = int(lengths.max())
            min_len = int(lengths.min())
            avg_len = float(lengths.mean())
        else:
            max_len = min_len = 0
            avg_len = 0.0
            
        col_summary = {
            "name": col,
            "type": infer_dtype(df[col]),
            "nulls": int(df[col].isna().sum()),
            "value_stats": {
                "num_values": int(num_values),
                "max_length": max_len,
                "min_length": min_len,
                "avg_length": round(avg_len, 2)
            }
        }
        summary["columns"].append(col_summary)

        # Distinct values per column (sample only a few)
        summary["values_sample"][col] = (
            col_data.unique()[:distinct_sample_size].tolist()
        )

    return summary

def process_csv_folder(input_folder, output_json: str, ground_truth_csv: str):
    all_summaries = []
    seen_schemas = set()

    target_tables = None
    target_tables = get_target_tables(ground_truth_csv)

    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    for file in csv_files:
        filename = os.path.basename(file)
        # if target_tables and filename not in target_tables:
        if filename not in target_tables:
            continue

        summary = summarize_csv(file)
        if summary is None:
            continue
        all_summaries.append(summary)
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=4, ensure_ascii=False)

    print(f"Schema with size {len(all_summaries)} summaries written to {output_json}")

data_folder = "path/to/tables/csv_files"
ground_truth_csv = "path/to/ground_truth.csv"
output_json = ""

tables = get_target_tables(ground_truth_csv)
process_csv_folder(
    data_folder,
    output_json=output_json,
    ground_truth_csv=ground_truth_csv
)
