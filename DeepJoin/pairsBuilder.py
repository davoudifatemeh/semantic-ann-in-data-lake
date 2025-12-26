import json
import random
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
from repositoryLoader import RepositoryLoader, column_to_text
from config import ANNOTATED, POS_PAIRS_OUTPUT, SPLIT_OUTPUT_PREFIX, SCHEMA, HEADER_INFO_WITH_SYNS, ANNOTATE_CORRUPTION, ANNOTATE_WITHOUT_HEADER

def col_to_text_with_sa(col: dict, col_mapping: json, include_header: bool) -> str:
    if col["corrupted_header"] is not None:
        print(col["file_name"], col["corrupted_header"])
        header = col["corrupted_header"].strip()
        annotation = col_mapping.get(col["file_name"], {}).get(col["corrupted_header"], "")
        print("Annotation:", annotation)
        if annotation:
            header = annotation
    else:
        header = col["header"].strip()
        annotation = col_mapping.get(col["file_name"], {}).get(col["header"], "")
        print("header:", header, " | Annotation:", annotation)
        if annotation:
            header = annotation
    sample_values = col.get("sample_values", [])
    n = col.get("num_values")
    if n > 0: # detailed info available
        title = col.get("title", "").strip()
        table = col.get("file_name", "").strip()
        num_values = col.get("num_values", 0)
        avg_length = col.get("avg_length", 0)
        min_length = col.get("min_length", 0)
        max_length = col.get("max_length", 0)
        text_parts = []

        if title:
            text_parts.append(title)
        text_parts.append(
            f"{header} contains {num_values} values "
            f"(min={min_length}, max={max_length}, avg={avg_length:.1f}):"
        )
        if sample_values:
            text_parts.append(", ".join(map(str, sample_values)))            
        return " ".join(text_parts)

class PairsBuilder:
    def __init__(self, annotation_file, use_annotation=False, include_header=True,
                  corrupt_header=False, corrupt_type=1, column_mapping=None):
        # annotation_file : str -> JSON annotated file or CSV ground truth file.
        
        self.use_annotation = use_annotation
        self.include_header = include_header
        self.corrupt_header = corrupt_header
        self.corrupt_type = corrupt_type
        self.annotation_file = annotation_file
        self.column_mapping = column_mapping
        ext = annotation_file.split(".")[-1].lower()
        self.mode = "json" if ext == "json" else "csv"

        if self.mode == "json":
            self.loader = RepositoryLoader(annotation_file)
            self.columns = self.loader.get_columns()
            self.groups = defaultdict(list) # group by semantic annotation or header
            for col in self.columns:
                self.groups[col["header"]].append(col) # or group by semantic_annotation

        else:  # CSV mode
            self.ground_truth = pd.read_csv(annotation_file)
            header_info = None
            if self.use_annotation:
                header_info = HEADER_INFO_WITH_SYNS
            self.loader = RepositoryLoader(annotation_file, SCHEMA, header_info, corrupt_header=self.corrupt_header, corrupt_type=self.corrupt_type)
            self.columns = self.loader.get_columns()
            # build lookup for quick access
            self.lookup = {
                (col["file_name"], col["header"]): col
                for col in self.columns
            }
            self.groups = dict(tuple(self.ground_truth.groupby("query_column")))


    def build_pairs(self):
        # Create positive pairs from ground-truth file
        pairs = []
        if self.mode == "json":
            for fam, cols in self.groups.items():
                if len(cols) < 2:
                    continue
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        textA = column_to_text(cols[i])
                        textB = column_to_text(cols[j])
                        pairs.append((textA, textB, fam.lower()))
        else:  # CSV mode
            for _, row in self.ground_truth.iterrows():
                q_table, q_col = row["query_table"], row["query_column"]
                c_table, c_col = row["candidate_table"], row["candidate_column"]

                q_col_meta = self.lookup.get((q_table, q_col))
                if q_col_meta:
                    if (ANNOTATE_CORRUPTION != 0) or ANNOTATE_WITHOUT_HEADER:
                        col1_text = col_to_text_with_sa(q_col_meta, self.column_mapping, self.include_header)
                    else:
                        col1_text = column_to_text(q_col_meta, self.use_annotation, self.include_header)
                else:
                    col1_text = str(q_col)

                c_col_meta = self.lookup.get((c_table, c_col))
                if c_col_meta:
                    if (ANNOTATE_CORRUPTION != 0) or ANNOTATE_WITHOUT_HEADER:
                        col2_text = col_to_text_with_sa(c_col_meta, self.column_mapping, self.include_header)
                    else:
                        col2_text = column_to_text(c_col_meta, self.use_annotation, self.include_header)
                else:
                    col2_text = str(c_col)
                pairs.append((col1_text, col2_text))
        print(f"PairsBuilder: Generated {len(pairs)} pairs")
        return pairs

    
    def save_all_pairs(pairs, out_file="positive_pairs.jsonl", fmt="jsonl"):
        if fmt == "jsonl":
            with open(out_file, "w", encoding="utf-8") as f:
                for p in pairs:
                    f.write(json.dumps(p) + "\n")
        elif fmt == "json":
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(pairs, f, indent=2)
        else:
            raise ValueError("Unsupported format: choose 'jsonl' or 'json'")

def split_pairs(pairs, train_ratio=0.9, seed=42):
    random.seed(seed)
    split_idx = int(len(pairs) * train_ratio)
    print(f"Splitting {len(pairs)} pairs at index {split_idx} for train/test.")
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    print(f"Split {len(pairs)} pairs into {len(train_pairs)} train and {len(test_pairs)} test pairs.")
    random.shuffle(train_pairs)
    random.shuffle(test_pairs)
    return train_pairs, test_pairs

def shuffle_pairs(pairs, seed=42):
    random.seed(seed)
    random.shuffle(pairs)
    return pairs


def save_split(split_pairs, split_name= "test", out_prefix="pairs", fmt="jsonl"):
    if fmt == "jsonl":
        with open(f"{out_prefix}_{split_name}.jsonl", "w", encoding="utf-8") as f:
            for p in split_pairs:
                f.write(json.dumps(p) + "\n")
        print(f"Saved {len(split_pairs)} pairs to {out_prefix}_{split_name}.jsonl")

    elif fmt == "json":
        with open(f"{out_prefix}_{split_name}.json", "w", encoding="utf-8") as f:
            json.dump(split_pairs, f, indent=2)
        print(f"Saved {len(split_pairs)} pairs to {out_prefix}_{split_name}.json")

    else:
        raise ValueError("Unsupported format: choose 'jsonl' or 'json'")