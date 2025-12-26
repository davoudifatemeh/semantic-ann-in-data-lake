import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import torch
from config import ANNOTATED, SCHEMA, HEADER_INFO_WITH_SYNS

class RepositoryLoader:
    def __init__(self, annotated_file: str, csv_schema: str, header_info=None, corrupt_header=False, corrupt_type=1):
        # Load metadata from JSON (rich) or CSV (ground-truth pairs).

        self.columns = []
        self.rich_metadata = False  # Flag to signal if stats/context exist
        self.corrupt_header = corrupt_header
        self.corrupt_type = corrupt_type
        self.csv_schema = csv_schema
        self.header_info = header_info

        ext = os.path.splitext(annotated_file)[-1].lower()
        if ext == ".json":
            self._load_json(annotated_file)
            self.rich_metadata = True
        elif ext == ".csv":
            self._load_csv(annotated_file, csv_schema, header_info)
        else:
            raise ValueError(f"Unsupported annotation file type: {ext}")

    def _load_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        for table in metadata:
            file_name = table["file_name"]
            file_path = table.get("file_path", "")
            title = table.get("title", "")
            for col in table["columns"]:
                self.columns.append({
                    "file_name": file_name,
                    # "file_path": file_path,
                    "title": title,
                    "header": col["header"],
                    "num_values": col.get("num_values", 0),
                    "num_values": col.get("value_stats", {}).get("num_values", 0),
                    "avg_length": col.get("value_stats", {}).get("avg_length", 0),
                    "min_length": col.get("value_stats", {}).get("min_length", 0),
                    "max_length": col.get("value_stats", {}).get("max_length", 0),
                    "sample_values": col.get("values_sample", []),
                    "semantic_annotation": col.get("semantic_annotation", "")
                })

    def _load_csv(self, path: str, csv_schema: str =  SCHEMA, header_info: str = None):
        df = pd.read_csv(path)
        with open(csv_schema, "r", encoding="utf-8") as f:
            csv_schema = json.load(f)
        # build lookup for fast random access
        summary_lookup = {}
        for table in csv_schema:
            table_name = table["file"].strip()
            if table_name not in summary_lookup:
                summary_lookup[table_name] = {}
            for col in table["columns"]:
                entry = {
                    "value_stats": col.get("value_stats", {}),
                    "sample_values": table.get("values_sample", {}).get(col["name"], [])
                }
           
                for i in range(1, 4):
                    ckey = f"corrupt_{i}"
                    if ckey in col:
                        entry[ckey] = col[ckey]
                summary_lookup[table_name][col["name"]] = entry 

        header_info_lookup = {}
        if header_info is not None:
            with open(header_info, "r", encoding="utf-8") as f:
                header_data = json.load(f)
            for table in header_data:
                tname = table.get("table_name", "").strip()
                if not tname:
                    continue
                header_info_lookup[tname] = {
                    "table_description": table.get("table_description", ""),
                    "table_title": table.get("table_title", ""),
                    "columns": table.get("columns", {})
                }

        seen = set()
        for _, row in df.iterrows():
            for table_key, column_key in [
            ("query_table", "query_column"),
            ("candidate_table", "candidate_column")]:
                table_name = str(row[table_key]).strip()
                col_name =  str(row[column_key]).strip()
                if (table_name, col_name) in seen:
                    continue
                seen.add((table_name, col_name))

                stats = summary_lookup.get(table_name, {}).get(col_name, {})
                
                if not stats:
                    continue

                value_stats = stats.get("value_stats", {})
                samples = stats.get("sample_values", [])

                corrupted_header = None
                if self.corrupt_header:
                    ckey = f"corrupt_{self.corrupt_type}"
                    corrupted_header = stats.get(ckey, None)

                # look up ai metadata for column if available
                if table_name in header_info_lookup:
                    table_info = header_info_lookup.get(table_name, {})
                    table_title = table_info.get("table_title", "")
                    table_desc = table_info.get("table_description", "")
                    table_columns = table_info.get("columns", {})
                    semantic_name = table_columns.get(col_name, "")

                self.columns.append({
                    "file_name": table_name,
                    "header": col_name,
                    "corrupted_header": corrupted_header,
                    "num_values": value_stats.get("num_values", 0),
                    "avg_length": value_stats.get("avg_length", 0),
                    "min_length": value_stats.get("min_length", 0),
                    "max_length": value_stats.get("max_length", 0),
                    "sample_values": samples,
                    # ai generated metadata:
                    "table_title": table_title if header_info is not None else "",
                    "table_description": table_desc if header_info is not None else "",
                    "semantic_annotation": semantic_name if header_info is not None else ""
                })
    
    def get_columns(self) -> List[Dict[str, Any]]:
        print(f"Reopsitory: Loaded {len(self.columns)} columns.")
        return self.columns

def column_to_text(col: dict, use_annotation, include_header) -> str:
    print(f"column_to_text: Processing column {col['file_name']}::{col['header']}")
    if col["corrupted_header"] is not None:
        header = col["corrupted_header"].strip()

    else:
        header = col["header"].strip()

    sample_values = col.get("sample_values", [])
    n = col.get("num_values")
    if n > 0: # detailed info available
        title = col.get("title", "").strip()
        table = col.get("file_name", "").strip()
        num_values = col.get("num_values", 0)
        avg_length = col.get("avg_length", 0)
        min_length = col.get("min_length", 0)
        max_length = col.get("max_length", 0)
        if use_annotation:
            table_title = col.get("table_title", "").strip()
            table_description = col.get("table_description", "").strip()
            header = col.get("semantic_annotation", "").strip()

        text_parts = []
        if include_header:
            if title:
                text_parts.append(title)

            text_parts.append(
                f"{header} contains {num_values} values "
                f"(min={min_length}, max={max_length}, avg={avg_length:.1f}):"
            )
            if sample_values:
                text_parts.append(", ".join(map(str, sample_values)))
        else:
            text_parts.append(
                f"{num_values} values "
                f"(min={min_length}, max={max_length}, avg={avg_length:.1f}):"
            )
            if sample_values:
                text_parts.append(", ".join(map(str, sample_values)))
            
        return " ".join(text_parts)