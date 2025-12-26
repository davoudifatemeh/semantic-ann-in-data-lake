import os
from openai import OpenAI
import pandas as pd
from typing import Iterator, Tuple, List
import time
import json
from utils.config import config, RED, GREEN, YELLOW, RESET
from utils.ollama_health_check import check_connection
from utils.json_parser import json_parser, wordnet_mapper


class DataLoader:
    def __init__(self, sep: str = ','):
        self.sep = sep
        self.data_dir = config.data_dir

    def load_table(self, table_name) -> pd.DataFrame:
        """return table dataframe"""
        df = pd.DataFrame()
        if not (table_name.endswith(".csv") or table_name.endswith(".tsv")):
            print(f"{YELLOW}Not in Corrected Format.{RESET}")
            return
        
        file_path = self.data_dir + table_name
        print(f"\t{file_path} ...")
        if not os.path.exists(file_path):
            print(f"{YELLOW}\tFile {table_name} does not exist, skipping.{RESET}")
            return

        sep = "\t" if table_name.endswith(".tsv") else self.sep
        try:
            df = pd.read_csv(file_path, sep=sep, dtype="string", low_memory=False)
            return df
        except Exception as e:
            print(f"{RED}Could not load {table_name}: {e}{RESET}")
            return

class HeaderIdentification:

    def __init__(self):
        self.api_key = config.API_key
        self.model = config.model
        self.num_sample_rows = config.num_sample_rows
        self.test_part = config.test_part
        self.corruption_flag = config.corruption_flag
        self.split_number = config.split_number

        self.data_loader = DataLoader()

        self.client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key=self.api_key  # A key is required by the OpenAI library, but the value is ignored by Ollama
        )

    def generate_prompt(self, df: pd.DataFrame, table_name: str) -> tuple[str, str]:
        """
        Generates the system instruction and user prompt for the LLM.
        Args:    - df: The pandas DataFrame (table) to analyze.
        Returns: - A tuple containing (system_instruction, user_prompt_content).
        """
        print("\tGenerating User Prompt ....")
        current_columns = df.columns.tolist()

        # Get sample data randomly and format it cleanly (Markdown table is very readable by LLMs)
        df = df.drop_duplicates()
        sampled_df = df.sample(n=min(len(df), self.num_sample_rows), random_state=42).fillna("")
        sample_data_markdown = sampled_df.to_markdown(index=False, headers=[])

        # System Instruction: Defines the LLM's role and rules
        system_instruction = (f"""
            You are an expert Data Analyst specializing in feature engineering and data interpretation.
            Your task is to analyze raw table data and propose highly descriptive, meaningful, human-readable column names. 
            
            You MUST strictly adhere to the following rules:
            1. **Output Format:*** Your entire response must be a *single, valid JSON object*, that adheres exactly to the required schema:
            [SCHEMA]
            {config.response_format if not config.no_header else config.response_format_no_header}
                - Fields `table_name`, `table_description`, `table_title`, `columns` are reqiuerd in the json output and *no additional fields* must be included. 
                - `table_name` contains name of the table you see in the message and you are analysing and its type is string.
                - `table_description` contains a short description of what this table is about and its type is string.
                - `table_title` contains a short phrase (up to four words) indicating what this table is about and its type is string.
                - `columns` is a dictionary which contains mappings of columns' names.
            2. **Mapping:** The JSON must map the `{"old_column_name" if not config.no_header else "column_n"}` (key) to the `suggested_meaningful_name` (value). It also contains 'table_name' which represent name of the table like {table_name} and 'table_description' which provide a short description of what table is about.
            3. **Naming Convention:** All suggested names must be in `snake_case` (e.g., `customer_id`, `transaction_amount`)
            4. **Completeness:** You must suggest a new name for *every* column provided in the input, and only those columns.
            """
        )

        # User Prompt: Provides the specific data context
        user_prompt_content = f"""
            Analyze the following data table and produce the corresponding output:

            [Table Name] 
            {table_name}"""
        
        if not config.no_header:
            user_prompt_content += f"""

            [Columns names]
            {current_columns}"""
        else:
            user_prompt_content += f"""

            [Columns count]
            {len(current_columns)}"""
            print(f"\t{YELLOW}Assuming columns have no headers.{RESET}")

        user_prompt_content += f"""

            [Sample Data (First {self.num_sample_rows} Rows)]
            {sample_data_markdown}
            """

        return system_instruction, user_prompt_content

    def _llm_request(self, system_instruction: str, user_prompt: str) -> dict:
        """Send request to LLM and return the response JSON."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ]
            )
            raw_content = response.choices[0].message.content
            print(f"\n\tRaw Content: {raw_content}\n")
            return raw_content
        except Exception as e:
            print(f"{RED}\tError during LLM request: {e}{RESET}")
            return None

    def identify_header(self) -> dict:
        print(f"Identifying table headers ...")
        i = 0
        answers = []
        table_names = []
        data_dir = config.data_dir

        if self.test_part:
            tables_df = pd.read_csv(config.ground_truth_file, dtype="string", low_memory=False)
            c_tables = tables_df['candidate_table'].iloc[self.split_number:].unique().tolist()
            q_tables = tables_df['query_table'].iloc[self.split_number:].unique().tolist()
            table_names.extend(c_tables)
            table_names.extend(q_tables)
            table_names = list(set(table_names))
            print(f"Processing {YELLOW}only test space tables{RESET}, total tables: {len(table_names)}")
        else:
            table_names = os.listdir(data_dir)
            print(f"Processing all tables, total tables: {len(table_names)}")

        if config.corruption_flag != 0 and not config.no_header:
            corruption_type = config.corruption_flag
            try:
                corrupted_schema = json.load(open(config.corruption_file))
                print(f"{GREEN}Loaded corrupted schema with {len(corrupted_schema)} successfully.{RESET}")
            except Exception as e:
                print(f"{RED}Error Loading Corrupted Schema: {e}{RESET}")
                return answers

        for table_name in table_names:
            _json = None

            print(f"Processing table: {table_name}")
            df = self.data_loader.load_table(table_name)
            if df is None:
                continue

            if df.empty:
                print(f"\t{YELLOW}Table {table_name} is empty, skipping.{RESET}")
                continue

            if config.corruption_flag != 0 and not config.no_header:
                # Apply corruption to column names
                corrupted_columns = {}
                for entry in corrupted_schema:
                    if entry['file'] == table_name:
                        for col in entry['columns']:
                            if corruption_type == 1:
                                corrupted_columns[col['name']] = col['corrupt_1']
                            elif corruption_type == 2:
                                corrupted_columns[col['name']] = col['corrupt_2']
                            elif corruption_type == 3:
                                corrupted_columns[col['name']] = col['corrupt_3']
                        break
                # Rename columns in the dataframe
                df.rename(columns=corrupted_columns, inplace=True)

            system_instruction, user_prompt = self.generate_prompt(df, table_name)
            i += 1
            raw_content = self._llm_request(system_instruction, user_prompt)

            columns_names = df.columns.tolist()
            _json = json_parser.extract_clean_json(raw_content, columns_names, table_name)
            if _json:
                print(f"\n\tJSON: {_json}")
                answers.append(_json)
            else:
                raw_content = self._llm_request(system_instruction, user_prompt)
                _json = json_parser.extract_clean_json(raw_content, columns_names, table_name)
                if _json:
                    print(f"\n\tJSON (2nd Attempt): {_json}")
                    answers.append(_json)
                else:
                    print(f"{RED}\tFailed to extract valid JSON after 2 attempts for table {table_name}.{RESET}")
            print(f"\tReceived response for table {table_name}")
        return answers

if __name__ == "__main__":
    stime = time.time()
    header_identifier = HeaderIdentification()

    if check_connection(header_identifier.client):
        results = header_identifier.identify_header()
        if config.corruption_flag != 0 and not config.no_header:
            output_path = os.path.join(config.output_dir, f"sa_results_corruption_{config.corruption_flag}.json")
        elif config.no_header:
            output_path = os.path.join(config.output_dir, "sa_results_no_header.json")
        else:
            output_path = os.path.join(config.output_dir, "sa_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        
        etime = time.time()
        ttime = (etime - stime) / 60
        print(f"{GREEN}Results saved to {output_path}{RESET}\nTotal Execution Time: {ttime:.2f}")
    else:
        print("Cannot proceed without a connection to the Ollama API.")

    stime = time.time()
    if os.path.exists(config.output_dir + "sa_results.json") and config.corruption_flag == 0 and not config.no_header:
        print(f"WrdNet Mapping Started ...")
        data = json.load(open(config.output_dir + "sa_results.json"))
        new_data, mapper = wordnet_mapper.process_json_with_wordnet(data)
        output_path = os.path.join(config.output_dir, "sa_results_wordnet_mapped.json")
        with open(output_path, "w") as f:
            json.dump(new_data, f, indent=4)
        
        mapper_path = os.path.join(config.output_dir, "wordnet_mapping.json")
        with open(mapper_path, "w") as f:
            json.dump(mapper, f, indent=4)
        etime = time.time()
        ttime = (etime - stime) / 60
        print(f"{GREEN}Results saved to {output_path} and {mapper_path}{RESET}\nTotal Execution Time: {ttime:.2f}")