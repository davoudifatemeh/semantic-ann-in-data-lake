import os
import json
import pandas as pd

from datetime import datetime
from collections import Counter, defaultdict

from config import config

class DataLoader:
    """
    Handles loading and preprocessing of CSV/TSV files, with optional JSON metadata.
    It generates a standardized data lake representation (list of dictionaries)
    including basic stats and initial LLM-based descriptions.
    """
    def __init__(self):
        config.create_directory_if_not_exists(config.local_output_directory)
        config.create_directory_if_not_exists(config.global_output_directory)

    def _format_date(self,
        date_str: str
    ) -> str | None:
        """
        Private helper method to convert date strings to standardized format.

        This method takes a string date_str and attempts to convert it to
        a standardized format of 'YYYY-MM-DDTHH:MM' using various date formats.
        If the conversion is successful, it returns the converted date string.
        Otherwise, it returns None.

        This method is used internally by the DataLoader to convert date columns
        to a standardized format. The supported date formats are:
            - YYYY-MM-DDTHH:MM (ISO 8601 format)
            - YYYY-MM-DD (date only)
            - YYYY-MM (date only, with month as a number)
            - YYYY-QX (quarterly dates, where X is 1-4)
            - YYYY-SX or YYYY-HX (semester dates, where X is 1 or 2)
            - YYYYMMDD (date only, with month and day as numbers)
            - MM/YYYY or MM-YYYY (date only, with month as a number)
            - Jan 2023, January 2023, etc. (textual dates)

        If the input date string does not match any of the supported formats,
        it returns None.
        """
        if not date_str or pd.isna(date_str):
            return None

        date_str = str(date_str).strip()

        try:
            # Handle quarterly dates (e.g., 2024-Q3)
            if 'Q' in date_str and len(date_str.split('-Q')) == 2 and date_str.split('-Q')[0].isdigit() and date_str.split('-Q')[1].isdigit():
                year, quarter = map(int, date_str.split('-Q'))
                month = (quarter - 1) * 3 + 1
                dt = datetime(year, month, 1)
                return dt.strftime('%d %B %Y')
                # return dt.strftime('%Y-%m-%dT%H:%M')

            # Handle YYYY-MM (e.g., 2005-03)
            if len(date_str) == 7 and date_str[4] == '-':
                year, month = map(int, date_str.split('-'))
                if 1 <= month <= 12:
                    dt = datetime(year, month, 1)
                    return dt.strftime('%d %B %Y')
                    # return dt.strftime('%Y-%m-%dT%H:%M')

            # Handle semester dates (e.g., 2023-S1, 2023-H1)
            if any(s in date_str for s in ['-S1', '-S2', '-H1', '-H2']):
                parts = re.split(r'-[SH]', date_str)
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    year, semester = int(parts[0]), int(parts[1])
                    month = 1 if semester == 1 else 7
                    dt = datetime(year, month, 1)
                    return dt.strftime('%d %B %Y')
                    # return dt.strftime('%Y-%m-%dT%H:%M')

            # Handle YYYYMMDD (e.g., 20230101)
            if len(date_str) == 8 and date_str.isdigit():
                year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
                if 1 <= month <= 12 and 1 <= day <= 31:
                    dt = datetime(year, month, day)
                    return dt.strftime('%d %B %Y')
                    # return dt.strftime('%Y-%m-%dT%H:%M')

            # Handle MM/YYYY or MM-YYYY
            if isinstance(date_str, str) and len(date_str) == 7 and (date_str[2] == '/' or date_str[2] == '-'):
                try:
                    if date_str[2] == '/':
                        month_str, year_str = date_str.split('/')
                    else:
                        month_str, year_str = date_str.split('-')

                    if month_str.isdigit() and year_str.isdigit():
                        month = int(month_str)
                        year = int(year_str)
                        if 1 <= month <= 12:
                            dt = datetime(year, month, 1)
                            return dt.strftime('%d %B %Y')
                            # return dt.strftime('%Y-%m-%dT%H:%M')
                except ValueError:
                    pass

            # Handle textual dates (e.g., "Jan 2023", "January 2023")
            if isinstance(date_str, str) and any(month in date_str for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
                try:
                    for fmt in ['%b %Y', '%B %Y', '%b, %Y', '%B, %Y', '%d-%m-%y', '%b %d' ]:
                        try:
                            dt = datetime.strptime(date_str, fmt)
                            return dt.strftime('%d %B %Y')
                            # return dt.strftime('%Y-%m-%dT%H:%M')
                        except ValueError:
                            continue
                except Exception:
                    pass

            # Try different date formats
            for fmt in [
                '%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d', '%d-%m-%Y', '%Y',
                '%d.%m.%Y', '%m/%d/%Y', '%Y.%m.%d', '%d %b %Y', '%d %B %Y',
                '%b %d, %Y', '%B %d, %Y', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M',
                '%d/%m/%Y %H:%M', '%d/%m/%Y %H:%M:%S'
            ]:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%d %B %Y')
                    # return dt.strftime('%Y-%m-%dT%H:%M')
                except ValueError:
                    continue

            return None # If none of the formats match
        except Exception as e:
            print(config.add_timestamp(config.color_text(f"Error formatting date '{date_str}': {e}", "red")))
            return None

    def load_semantics_and_stats(self) -> list[dict] | None:
        """
        Loads schema and statistics from provided JSON files.

        Args:
            schema_file (str): Path to the JSON file containing schema information.
            stats_file (str): Path to the JSON file containing statistics information.
        Returns:
            dict: A dictionary mapping table names to their schema and statistics.
        """
        output_directory = os.path.join(config.local_output_directory, "stats_and_semantics_total.json")
        data_lake = config.exist_json(output_directory)
        if data_lake:
            print(config.add_timestamp(config.color_text(f"Data Lake has {len(data_lake)} tables", "green")))
            return data_lake

        data_lake = []
        stats_file = config.stats_file
        semantics_file = config.semantics_file
        total_columns = 0
        # Load statistics
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats_data = json.load(f)
            for table in stats_data:
                table_name = table['file']
                data_lake.append({
                    "file_name": table_name,
                    "llm_description": "",
                    "title": "",
                    "row_count": table.get("row_count", 0),
                    "columns": table.get("columns", [])
                })
                total_columns += len(table.get("columns", []))
                values_sample = table.get("values_sample", [])
                if values_sample:
                    for col_name, col_sample in values_sample.items():
                        for col in data_lake[-1]["columns"]:
                            if col['name'] == col_name:
                                col.setdefault('values_sample', []).extend(col_sample)
                            break
        except Exception as e:
            print(config.add_timestamp(config.color_text(f"Error loading statistics from {stats_file}: {e}", "red")))
            return None
        print(f"Total columns: {total_columns}")
        # Load semantics
        table_names = {table['file_name'] for table in data_lake}
        try:
            with open(semantics_file, 'r', encoding='utf-8') as f:
                semantics_data = json.load(f)
            for table in semantics_data:
                table_name = table['table_name']
                if table_name in table_names:
                    for dl_table in data_lake:
                        if dl_table['file_name'] == table_name:
                            dl_table['llm_description'] = table.get('table_description', "")
                            dl_table['title'] = table.get('table_title', "")
                            for col in dl_table['columns']:
                                set_annotation = False
                                col_name = col['name']
                                for _name, _sem in table.get('columns', {}).items():
                                    if _name == col_name:
                                        if "date" in _sem.lower():
                                            col.update({'type': 'date'})
                                            formatted_dates = []
                                            for date_val in col.get('values_sample', []):
                                                formatted_date = self._format_date(date_val)
                                                if formatted_date:
                                                    formatted_dates.append(formatted_date)
                                            if formatted_dates: 
                                                col.update({'values_sample': formatted_dates})
                                        col.update({'semantic_annotation': _sem})
        except Exception as e:
            print(config.add_timestamp(config.color_text(f"Error loading semantics from {semantics_file}: {e}", "red")))
            return None

        try:
            with open(output_directory, 'w', encoding='utf-8') as f:
                json.dump(data_lake, f, indent=2)
            print(config.add_timestamp(config.color_text(f"Combined semantics and statistics saved to: {output_directory}", "green")))
        except Exception as e:
            print(config.add_timestamp(config.color_text(f"Error saving combined semantics and statistics to {output_directory}: {e}", "red")))
            return None
        return data_lake

data_loader = DataLoader()