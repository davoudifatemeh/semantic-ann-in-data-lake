import json
import re
from typing import List, Dict, Any, Tuple, Optional
from utils.config import config, RED, RESET, GREEN, YELLOW
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import pandas as pd

class JsonParser:

    def __init__(self):
        self.json_sec = None

    def find_target_section(self, data: Any) -> Optional[Dict[str, Any]]:
        """
        Recursively searches dictionary data for an object containing all required keys.
        """
        if not isinstance(data, dict):
            return None

        # Check if the current dictionary contains all required keys
        if config.required_keys.issubset(data.keys()) and isinstance(data['table_name'], str):
            return data

        # If not found at the current level, search its children
        for value in data.values():
            if isinstance(value, dict):
                # Recursively call on the nested dictionary
                result = self.find_target_section(value)
                if result is not None:
                    return result
        return None

    def extract_clean_json(self, llm_response_text: str, columns_names: list, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Extracts the relevant structured data (table_name, table_description, columns) from a 
        raw LLM response, handling verbose text, markdown fences, and deeply nested JSON structures.

        Args:    - response_text: The raw string content from the LLM response.
        Returns: - A dictionary containing only the required fields, or None if extraction fails.
        """
        
        # Regular Expression to Extract JSON Block
        # Looks for content inside markdown (```json...```) or the first standalone JSON object
        match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```|(^\{[\s\S]*\}+\s*$)|```(^\{[\s\S]*\}+\s*$)```", llm_response_text)
        if not match:
            print(f"{RED}\tCould not find a recognizable JSON block.{RESET}")
            return None
        else:
            # Get the captured JSON string (group 1 is markdown, group 2 is standalone)
            json_string = match.group(1) or match.group(2) or match.group(2)
            
        # Parse the JSON string
        try:
            raw_data = json.loads(json_string.strip())
        except json.JSONDecodeError as e:
            print(f"{RED}\tJSON Decoding Error: {RESET}{e} while parsing: {json_string[:100]}...")
            return None

        # Search for the required structure using recursion
        target_data = self.find_target_section(raw_data)
            
        if not target_data:
            print(f"{RED}\tThe required keys {config.required_keys} were not found in any nested dictionary.{RESET}")
            return None

        # Final cleaning: extract ONLY the required keys and validate 'columns'
        cleaned_data = {}

        for key in config.required_keys:
            cleaned_data[key] = target_data[key]

        if not isinstance(cleaned_data.get("columns"), dict):
            print(f"{RED}\t'columns' validation failed{RESET}")
            return None
        
        cleaned_data["table_name"] = table_name # Ensure table_name is correct

        if config.no_header:
            if len(columns_names) != len(cleaned_data['columns']):
                print(f"{YELLOW}\tColumn count mismatch between CSV and extracted JSON. CSV columns: {len(columns_names)}, JSON columns: {len(cleaned_data['columns'])}{RESET}")
                return False
            orig_keys = list(cleaned_data["columns"].keys())
            cleaned_data["columns"] = {
                columns_names[idx]: cleaned_data["columns"][col_key]
                for idx, col_key in enumerate(orig_keys)
            }
            print(f"{GREEN}\tColumn names adjusted based on CSV header.{RESET}")

        self.json_sec = cleaned_data
        print(f"{GREEN}\tJSON Extracted.{RESET}")
        return cleaned_data

class WordNetMapper:
    def __init__(self):
        self.mapped_names = {}

    def get_wordnet_synonyms(self, word: str) -> List[str]:
        """
        Retrieves a list of unique synonyms for a given word using WordNet.
        
        Args:
            word: The word to find synonyms for.
            
        Returns:
            A list of unique synonym lemmas (including the original word if found).
        """
        synonyms = set()
        normalized_word = word.lower().replace(' ', '_').replace('-', '_')

        try:
            for syn_list in wn.synonyms(normalized_word):
                if not syn_list:
                    continue
                for syn in syn_list:
                    if not syn in synonyms:
                        synonyms.add(syn)
        except Exception as e:
            print(f"{RED}\tWordNet lookup error for '{word}': {e}{RESET}")

        if word.lower() not in synonyms:
            synonyms.add(word.lower())
        
        print(f"WordNet Synonyms for '{word}': {synonyms}")
        return list(synonyms)

    def process_json_with_wordnet(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Processes the 'columns' to group synonyms using WordNet.
        For each unique column *value* (the human-readable description):
        1. Check if the value is already a synonym of a term we've processed. If so, map it.
        2. Otherwise, check WordNet for synonyms.
        3. If WordNet finds synonyms, the original value becomes the canonical term (key)
        and the list of synonyms is stored.
        4. If no meaningful synset is found, the value is kept as a standalone key.

        Args: data: The list of JSON objects (table definitions).

        Returns:
            A tuple containing:
            - The updated list of JSON objects with new column value mappings.
            - A dictionary representing the final synonym mapping group.
        """
        # { "canonical_term": [ "synonym1", "synonym2", ... ], ... }
        canonical_groups: Dict[str, List[str]] = {}
        
        # { "term": "canonical_term", ... }
        term_to_canonical: Dict[str, str] = {}
        
        # 1. Build the canonical_groups and term_to_canonical maps
        for table_def in data:
            print(f"Processing table: {table_def.get('table_name', 'Unknown')}")
            for original_key, term in table_def['columns'].items():
                if not isinstance(term, str):
                    print(f"{YELLOW}Skipping non-string term: {term}{RESET}")
                    continue
                # Standardize term for dictionary lookups
                term = term.lower()
                if term in term_to_canonical:
                    # Term already processed
                    print(f"{YELLOW}Term '{term}' already mapped to canonical term.{RESET}")
                    continue
                # Check if this term is a synonym of a previously found term
                found_group = False
                for canonical, synonyms in canonical_groups.items():
                    if term in synonyms:
                        # Found a match! The existing canonical term is the right group
                        term_to_canonical[term] = canonical
                        found_group = True
                        break
                if found_group:
                    print(f"{YELLOW}Term '{term}' mapped to existing canonical term.{RESET}")
                    continue
                # Term not found in existing groups, check WordNet
                synonyms = self.get_wordnet_synonyms(term)
                 
                if len(synonyms) > 1 and term in synonyms:
                    # The current term becomes the new canonical term.
                    canonical_groups[term] = synonyms
                    term_to_canonical[term] = term
                    for syn in synonyms:
                        if syn not in term_to_canonical:
                            term_to_canonical[syn] = term
                else:
                    canonical_groups[term] = [term]
                    term_to_canonical[term] = term

            print(f"Table '{table_def.get('table_name', 'Unknown')}' processed.")
        # 2. Apply the new mappings back to the JSON structure
        updated_data = json.loads(json.dumps(data)) 
        final_synonym_map = {} 
        for table_def in updated_data:
            new_columns = {}
            for original_key, term in table_def['columns'].items():
                if not isinstance(term, str):
                    print(f"{YELLOW}Skipping non-string term: {term}{RESET}")
                    continue
                term = term.lower()
                canonical_term = term_to_canonical.get(term, term) # Get the mapped or self-mapped canonical term
                new_columns[original_key] = canonical_term
            table_def['columns'] = new_columns

        # 3. final key-value map (canonical term -> list of synonyms)
        for canonical, synonyms in canonical_groups.items():
            mapped_terms = [t for t, c in term_to_canonical.items() if c == canonical]
            final_synonym_map[canonical] = list(set(mapped_terms))

        return updated_data, final_synonym_map

json_parser = JsonParser()
wordnet_mapper = WordNetMapper()