from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, List
from colorama import Fore, Style
import os
import time
import json

@dataclass
class Config:
    """Global configuration"""
    app_name: str = "webtable" # Options: "wiki", "webtable", "ecb"
    local_output_directory: str = f"/SemLink/output/{app_name}/"
    global_output_directory: str = f"/output/semlink/{app_name}/"
    stats_file: str = f"/data/{app_name}/{app_name}_csv_schema.json"
    semantics_file: str = f"/SemanticAnnotation/output/{app_name}/sa_results.json"
    query_col_file: str = f"/data/{app_name}/query_columns_spec.csv"
    candidate_col_file: str = f"/data/{app_name}/{app_name}_join_ground_truth.csv"
    ground_truth_file: str = f"/data/{app_name}/{app_name}_join_ground_truth.csv"
    embeddings_file: str = f"/SemLink/output/{app_name}/embeddings.json"
    query_col_spec = True  # Whether to use query column specification
    OPENAI_API_KEY: str = "API_KEY"  # OpenAI API Key
    semnatic_API_key: str = 'ANY'  # Not used with Ollama, but required by OpenAI library
    num_sample_rows: int = 10  # Number of sample rows to include in the
    embedding_model: str = "text-embedding-3-small"  # OpenAI embedding model
    top_k_neighbors: List(int) = field(default_factory=lambda: [5, 10, 25])  # Different k values for nearest neighbors
    distance_threshold: List(float) = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4]) # Distance threshold for joinable columns

    def color_text(self, text: str, color: str = 'white') -> str:
        """
        Colors text using colorama colors.

        Args:
            text (str): Text to be colored.
            color (str): Color name to use, defaults to 'white'.
                        Valid colors: 'black', 'red', 'green', 'yellow', 'blue',
                        'magenta', 'cyan', 'white'.

        Returns:
            str: Colored text string with reset style appended.
        """
        colors = {
            'black': Fore.BLACK,
            'red': Fore.RED,
            'green': Fore.GREEN,
            'yellow': Fore.YELLOW,
            'blue': Fore.BLUE,
            'magenta': Fore.MAGENTA,
            'cyan': Fore.CYAN,
            'white': Fore.WHITE
        }
        color_code = colors.get(color.lower(), Fore.WHITE)
        return f"{color_code}{text}{Style.RESET_ALL}"

    def add_timestamp(self, text: str) -> str:
        """
        Adds a timestamp before the provided text.

        Args:
            text (str): Text to prepend timestamp to.

        Returns:
            str: Text with timestamp prepended in format [HH:MM:SS].
        """
        timestamp = time.strftime("[%H:%M:%S]")
        return f"{timestamp} {text}"

    def create_directory_if_not_exists(self, path: str):
        """
        Creates a directory if it does not already exist.

        Args:
            path (str): The path to the directory to create.
        """
        os.makedirs(path, exist_ok=True)
        print(self.add_timestamp(self.color_text(f"Ensured directory exists: {path}", "green")))

    def load_datalake_json(self) -> list[dict]:
        """
        Loads a standardized data lake list of dictionaries from a JSON file.

        Args:
            json_file_path (str): The full path to the data lake JSON file.

        Returns:
            list[dict]: The loaded list of dictionaries representing the data lake.
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data_lake_list = json.load(f)
            print(add_timestamp(color_text(f"Data lake JSON loaded from: {json_file_path}", "green")))
            return data_lake_list
        except FileNotFoundError:
            print(add_timestamp(color_text(f"Error: Data lake JSON file not found at {json_file_path}", "red")))
            return []
        except json.JSONDecodeError as e:
            print(add_timestamp(color_text(f"Error decoding JSON from {json_file_path}: {e}", "red")))
            return []
        except Exception as e:
            print(add_timestamp(color_text(f"Error loading data lake JSON from {json_file_path}: {e}", "red")))
            return []

    def save_csv(self, df: pd.DataFrame, file_name: str, output_directory: str | None = None):
        output_directory = output_directory or self.global_output_directory
        file_path = os.path.join(output_directory, file_name)
        try:
            df.to_csv(file_path, index=False)
            print(self.add_timestamp(self.color_text(f"Saved DataFrame to CSV: {file_path}", "green")))
        except Exception as e:
            print(self.add_timestamp(self.color_text(f"Error saving DataFrame to CSV at {file_path}: {e}", "red")))
        return

    def save_json(self, data: dict, file_name: str, output_directory: str | None = None):
        output_directory = output_directory or self.global_output_directory
        file_path = os.path.join(output_directory, file_name)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            print(self.add_timestamp(self.color_text(f"Saved JSON Data to: {file_path}", "green")))
        except Exception as e:
            print(self.add_timestamp(self.color_text(f"Error saving data to JSON at {file_path}: {e}", "red")))
        return

    def exist_json(self, file_path):
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding='utf-8') as f:
                    file_json = json.load(f)
                print(config.add_timestamp(config.color_text(f"Loaded file {file_path} successfully", "green")))
                return file_json
            except Exception as e:
                print(config.add_timestamp(config.color_text(f"Error Loading File from {file_path}: {e}", "red")))
                return None

    def exist_csv(self, file_path, to_list: bool = False):
        if os.path.exists(file_path):
            try:
                file_df = pd.read_csv(file_path)
                print(config.add_timestamp(config.color_text(f"Loaded file {file_path} successfully", "green")))
                if to_list:
                    return file_df.to_list()
                return file_df
            except Exception as e:
                print(config.add_timestamp(config.color_text(f"Error Loading File from {file_path}: {e}", "red")))
                return None


config = Config()