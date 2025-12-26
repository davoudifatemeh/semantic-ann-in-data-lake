from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    """Global configuration for Schema Extraction"""
    app_name: str = "webtable"  # Options: "wiki", "webtable", "ecb"
    data_dir: str = f"/data/{app_name}/tables/"
    output_dir: str = f"/SemLink/output/{app_name}/"
    API_key: str = 'ANY'  # Not used with Ollama, but required by OpenAI library
    model: str = "gpt-oss:20b"  # Ollama model name
    num_sample_rows: int = 10  # Number of sample rows to include in the
    test_part: bool = True  # If True, only process test tables
    ground_truth_file: str = f"/data/{app_name}/{app_name}_join_ground_truth.csv"
    corruption_flag: int = 3 # 1: corruption 1, 2: corruption 2, 3: corruption 3, 0: no corruption
    corruption_file: str = f"/data/{app_name}/{app_name}_csv_schema_corr.json"
    no_header: bool = False  # If True, assume columns have no headers
    split_number: int = 49339  # wiki: 1968, webtable: 49339, ecb: 54890
    
    # required keys for json response of Ollama
    required_keys = {"table_name", "table_description", "table_title", "columns"}
    response_format = """
{
  "json_schema":{
    "properties":{
        "table_name": "",
        "table_description": "",
        "table_title": "",
        "columns":{
            "old_column_name_1": "suggested_meaningful_name",
            "old_column_name_2": "suggested_meaningful_name",
            // ... all columns must be included here
        }
    }
}"""

    response_format_no_header = """
{
  "json_schema":{
    "properties":{
        "table_name": "",
        "table_description": "",
        "table_title": "",
        "columns":{
            "column_1": "suggested_meaningful_name",
            "column_2": "suggested_meaningful_name",
            // ... all columns must be included here
        }
    }
}"""

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

config = Config()