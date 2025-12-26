import os
import json
import time
import numpy as np
import pandas as pd
import faiss
import tiktoken
from tqdm import tqdm
from openai import OpenAI
from typing import Union
from config import config
from collections import defaultdict

class JoinDiscoverer:
    """
    This class provides a streamlined workflow to:
    1. Generate embeddings for a data lake's columns using an LLM.
    2. Calculate various distance metrics between these embeddings.
    """
    def __init__(self, openai_client: OpenAI):
        """
        Initializes the DataJoinerPipeline.

        Args:
            openai_client (OpenAI): An initialized OpenAI client instance.
        """
        self.eps = 1e-8
        self.openai_client = openai_client
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(config.add_timestamp(config.color_text(f"Warning: Could not load tiktoken tokenizer: {e}", "yellow")))

    def _truncate_text(self,
        text: str,
        max_tokens: int,
        model_name: str = config.embedding_model
    ) -> str:
        """
        Private helper method to truncate text to fit within the token limit for the embedding model.

        Args:
            text (str): The text to truncate.
            max_tokens (int): The maximum number of tokens allowed for the embedding model.
            model_name (str): The name of the embedding model to use.

        Returns:
            str: The truncated text, or the original text if it is already within the token limit.
        """
        if self.tokenizer:
            encoded_text = self.tokenizer.encode(text)
        else:
            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            encoded_text = encoding.encode(text)

        if len(encoded_text) > max_tokens:
            truncated_text = self.tokenizer.decode(encoded_text[:max_tokens])
            return truncated_text
        return text

    def _prepare_embedding_prompt(self, column_data: dict, table_description: str, max_tokens: int) -> Union[str, None]:
        """
        The prompt is built conditionally based on the available data and the specified mode.
        This method now combines information from semantic annotations, LLM descriptions,
        and column value statistics for a richer context.

        Args:
            column_data (dict): A dictionary containing all information for a single column.
            mode (str): The mode to use for preparing the embedding prompt.
                        Options: 'semantic_mode', 'header_mode'.
            max_tokens (int): The maximum number of tokens allowed for the prompt.

        Returns:
            Union[str, None]: The prepared prompt string, or None if no valid data is found.
        """
        # Extract data from the column dictionary
        header = column_data.get('header', column_data.get('name', ''))
        semantic_annotation = column_data.get('semantic_annotation', 'NA')
        value_stats = column_data.get('value_stats', {})
        attribute_examples = ", ".join(column_data.get('values_sample', []))

        # Extract specific statistics
        num_values = value_stats.get('num_values', 0)
        max_length = value_stats.get('max_length', 0)
        min_length = value_stats.get('min_length', 0)
        avg_length = value_stats.get('avg_length', 0.0)
        most_common_values = value_stats.get('most_common_values', [])

        text_parts = []
        if semantic_annotation != 'NA':

            prompt_line = f"The attribute with header: '{header}' has semantic annotation: '{semantic_annotation}'"
            if table_description:
                prompt_line += f" and belongs to a table which is described as: '{table_description}'"
            else:
                prompt_line += "."
            text_parts.append(prompt_line)

        # Add value examples if available
        if attribute_examples:
            text_parts.append(f"Examples of values for this attribute include: '{attribute_examples}'.")

        # Add column statistics if available
        if num_values > 0:
            text_parts.append(f"The dataset for column '{header}' contains {num_values} entries.")

        if max_length > 0 and min_length > 0:
            text_parts.append("Key statistics for the column:")
            text_parts.append(f"- Maximum value length: {max_length} characters.")
            text_parts.append(f"- Minimum value length: {min_length} characters.")
            text_parts.append(f"- Average value length: {avg_length:.1f} characters.")

        # Add most frequent values if available
        if most_common_values:
            text_parts.append("Top 20 most frequent values in the column:")
            text_parts.append(", ".join([str(v) for v in most_common_values[:20]]))

        # Join and truncate the final prompt
        final_prompt = "\n".join([part for part in text_parts if part])

        if final_prompt:
            return self._truncate_text(final_prompt, max_tokens)
        else:
            return None

    def _load_query_columns(self) -> list[str]:
        """
        Private helper method to load the query columns if specified in the config.

        Returns:
            list[str]: A list of query column full names, or an empty list if not specified.
        """
        query_columns = []
        if not os.path.exists(config.query_col_file):
            print(config.add_timestamp(config.color_text(f"Error: Query column spec file not found at {config.query_col_file}", "red")))
            return query_columns
        try:
            df_query_cols = pd.read_csv(config.query_col_file).drop_duplicates(keep='first')
            query_columns.append(f"{row['query_table']}:{row['query_column']}" for _, row in df_query_cols.iterrows())
            print(config.add_timestamp(config.color_text(f"Loaded query columns {len(df_query_cols)}", "green")))
        except Exception as e:
            print(config.add_timestamp(config.color_text(f"Error loading query columns: {e}", "red")))

        return query_columns  

    def _get_embedding(self, text: str, model: str) -> Union[list[float], None]:
        """
        Private helper method to get the embedding for a given text using the specified model.

        Args:
            text (str): The text for which to generate an embedding.
            model (str): The name of the OpenAI embedding model to use.

        Returns:
            Union[list[float], None]: The embedding for the text, or None if an error occurs.
        """
        try:
            response = self.openai_client.embeddings.create(
                input=[text],
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print(config.add_timestamp(config.color_text(f"Error getting embedding for text: '{text[:50]}...': {e}", "red")))
            return None

    def generate_embeddings(self,
        data_lake_list: list[dict] | str,
        embedding_model: str = config.embedding_model,
        output_directory: str = config.local_output_directory
    ) -> list[dict]:
        """
        Generates a list of embeddings for each column in the data lake representation.

        Args:
            data_lake_list (list[dict] | str): The list of dictionaries representing the data lake,
                                            as generated by DataLoader.load_semantics_and_stats.
            embedding_model (str): The name of the OpenAI embedding model to use.
            output_directory (str | None): If specified, the generated embeddings will be saved to a file
                                        in this directory, named "embeddings.json".

        Returns:
            list[dict]: A list of dictionaries, each containing the embedding for a column and its file name,
                        as well as the semantic annotation for the column (if available).
        """
        output_file = os.path.join(output_directory, "embeddings.json")
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding='utf-8') as f:
                    embeddings_list = json.load(f)
                print(config.add_timestamp(config.color_text(f"Loaded existing embeddings from: {output_file}", "green")))
                return embeddings_list
            except Exception as e:
                print(config.add_timestamp(config.color_text(f"Error loading existing embeddings from {output_file}: {e}", "red")))

        if not data_lake_list:
            print(config.add_timestamp(config.color_text("No data lake information provided for schema pruning.", "yellow")))
            return None
            
        if isinstance(data_lake_list, str):
            if not os.path.exists(data_lake_list):
                print(config.add_timestamp(config.color_text(f"Error: Data lake JSON file not found at {data_lake_list}", "red")))
            data_lake_list = load_datalake_json(data_lake_list)
        
        embeddings_list = []
        max_prompt_tokens = 8191 if 'text-embedding-3-large' in embedding_model else 2048

        total_columns = sum(len(file_info['columns']) for file_info in data_lake_list)
        success_count = 0
        prompt = []
        with tqdm(total=total_columns, desc=config.add_timestamp("Generating Embeddings")) as pbar:
            for file_info in data_lake_list:
                table_description = file_info['llm_description']
                for column_dict in file_info['columns']:
                    full_column_name = f"{file_info['file_name']}:{column_dict['name']}"

                    prompt_text = self._prepare_embedding_prompt(
                        column_data=column_dict,
                        table_description=table_description,
                        max_tokens=max_prompt_tokens
                    )

                    if prompt_text is None:
                        continue
                    prompt.append({
                        'column_name': full_column_name,
                        'prompt_text': prompt_text
                    })
                    embedding = self._get_embedding(prompt_text, model=embedding_model)

                    if embedding is not None:
                        embeddings_list.append({
                            'column_name': full_column_name,
                            'embedding': embedding,
                            'semantic_annotation': column_dict.get('semantic_annotation', 'NA')
                        })
                        success_count += 1
                        pbar.set_postfix(success=success_count)
                        
                    pbar.update(1)
                    time.sleep(0.1)

        print(config.add_timestamp(config.color_text(f"Finished generating embeddings for {len(embeddings_list)} columns.", "green")))
        print(config.add_timestamp(config.color_text(f"Finished preparing prompts for {len(prompt)} columns.", "green")))
        config.save_json(prompt, "prompts.json", config.local_output_directory)
        
        if output_directory:
            config.save_json(embeddings_list, "embeddings.json", config.local_output_directory)
        
        return embeddings_list

    def _compute_distances(self,
        embeddings: list[dict],
        k_neighbors: int = 50
    ) -> pd.DataFrame:
        """
        Compute cosine similarity, euclidean distance, and ANNS distances between all pairs of column embeddings.

        Args:
            embeddings (list[dict]): A list of dictionaries, each containing the embedding for a column and its file name,
                                     as well as the semantic annotation for the column (if available).
            k_neighbors (int): The number of nearest neighbors to consider for ANNS distance calculation.

        Returns:
            pd.DataFrame: A DataFrame containing the distances and similarities between all pairs of columns.
        """
        column_names = [d['column_name'] for d in embeddings]
        embedding_matrix = np.array([d['embedding'] for d in embeddings])

        # --- Calculate ANNS Distances (using FAISS) ---
        print(config.add_timestamp(config.color_text("Calculating ANNS distances...", 'cyan')))
        try:
            d = embedding_matrix.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(embedding_matrix.astype('float32'))
            k = min(k_neighbors, len(column_names))
            distances_anns, indices_anns = index.search(embedding_matrix.astype('float32'), k)

        except ImportError:
            print(config.add_timestamp(config.color_text("FAISS library not found. ANNS distances will be skipped.", "red")))
            distances_anns, indices_anns = None, None
        except Exception as e:
            print(config.add_timestamp(config.color_text(f"Error calculating ANNS distances: {e}. ANNS distances will be skipped.", "red")))
            distances_anns, indices_anns = None, None
        
        # Export distances
        distances = []
        for i in range(len(column_names)):
            for rank in range(1, k):  # skip rank 0 (self)
                j = indices_anns[i, rank]
                distances.append({
                    "column_1": column_names[i],
                    "column_2": column_names[j],
                    "distance_anns": distances_anns[i, rank]
                })
        sorted_distances = sorted(distances, key=lambda x: (x['distance_anns']))

        return pd.DataFrame(sorted_distances)

    def _extract_joinable_columns(self,
        df_distances: pd.DataFrame,
        dist_threshold: float = 0.2,
        dist_mode: str = 'anns'
    ) -> pd.DataFrame:
        """
        Extracts joinable column pairs based on a specified distance threshold.

        Args:
            df_distances (pd.DataFrame): A DataFrame containing the distances and similarities between all pairs of columns.
            dist_threshold (float): The maximum euclidean distance for two columns to be considered joinable.

        Returns:
            pd.DataFrame: A DataFrame containing the joinable column pairs.
        """
        joinable_df = df_distances[df_distances['distance_anns']**0.5 <= dist_threshold].copy()
        print(config.add_timestamp(config.color_text(f"Found {len(joinable_df)} joinable column pairs with distance <= {dist_threshold}.", "green")))

        joinable_df[['t1', 'c1']] = joinable_df['column_1'].str.split(':', expand=True)
        joinable_df[['t2', 'c2']] = joinable_df['column_2'].str.split(':', expand=True)
        final_joinable_df = joinable_df[['t1', 'c1', 't2', 'c2', 'distance_anns']]

        return final_joinable_df

    def _calculate_ndcg(self, df_distances: pd.DataFrame, ground_truth_map: dict, k: int) -> float:
            """
            Modular helper to calculate NDCG@k.
            Args:
                df_distances: DataFrame with 'column_1', 'column_2' and 'distance_anns'.
                ground_truth_map: Dictionary mapping query_column -> set(relevant_candidate_columns).
                k: The cutoff rank.
            Returns:
                float: The average NDCG score.
            """
            ndcg_scores = []

            grouped = df_distances.sort_values('distance_anns').groupby('column_1')
            
            for query, group in grouped:
                candidates = group['column_2'].values[:k]
                
                relevant_items = ground_truth_map.get(query, set())
                if not relevant_items:
                    continue
                    
                dcg = 0.0
                relevance_list = []
                for i, candidate in enumerate(candidates):
                    rel = 1 if candidate in relevant_items else 0
                    relevance_list.append(rel)
                    if rel:
                        dcg += 1.0 / np.log2(i + 2) # i=0 -> log2(2)
                
                num_relevant = len(relevant_items)
                
                idcg = 0.0
                
                for i in range(min(k, num_relevant)):
                    idcg += 1.0 / np.log2(i + 2)
                    
                if idcg > 0:
                    ndcg_scores.append(dcg / idcg)
                else:
                    ndcg_scores.append(0.0)

            return np.mean(ndcg_scores) if ndcg_scores else 0.0

    def _evaluation_report(self,
        df_joinable: pd.DataFrame,
        df_distances: pd.DataFrame,
        k: int,
        grounth_truth_file: str = config.ground_truth_file,
        query_col_spec: bool = config.query_col_spec
    ) -> dict:
        """
        Generates an evaluation report comparing the discovered joinable columns against ground truth.

        Args:
            df_joinable (pd.DataFrame): A DataFrame containing the discovered joinable column pairs.
            grounth_truth_file (str): The path to the ground truth CSV file.

        Returns:
            dict: A dictionary containing precision, recall, and F1-score.
        """
        if not os.path.exists(grounth_truth_file):
            print(config.add_timestamp(config.color_text(f"Error: Ground truth file not found at {grounth_truth_file}", "red")))
            return None
        df_ground_truth = pd.read_csv(grounth_truth_file)

        gt_map = defaultdict(set)
        for _, row in df_ground_truth.iterrows():
            q = row['query_table'] + ':' + row['query_column']
            c = row['candidate_table'] + ':' + row['candidate_column']
            gt_map[q].add(c)
            gt_map[c].add(q)

        total_actual_pairs = set(
            tuple(sorted([row['query_table'] + ':' + row['query_column'], row['candidate_table'] + ':' + row['candidate_column']]))
            for _, row in df_ground_truth.iterrows()
        )
        
        total_discovered_pairs = set(
            tuple(sorted([row['t1'] + ':' + row['c1'], row['t2'] + ':' + row['c2']]))
            for _, row in df_joinable.iterrows()
        )
        
        query_cols = set()
        if query_col_spec:
            df_query_col_spec = pd.read_csv(config.query_col_file)
            query_cols = set(row['query_table'] + ':' + row['query_column']
                for _, row in df_query_col_spec.iterrows())

        true_pairs = set(pair for pair in total_actual_pairs if pair[0] in query_cols)
        discovered_pairs = set(pair for pair in total_discovered_pairs if pair[0] in query_cols)
        true_positives = len(discovered_pairs.intersection(true_pairs))
        false_positives = len(discovered_pairs) - true_positives
        false_negatives = len(true_pairs) - true_positives

        precision = true_positives / (true_positives + false_positives + self.eps)
        recall = true_positives / (true_positives + false_negatives + self.eps)
        f1_score = 2 * (precision * recall) / (precision + recall + self.eps)

        eval_metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

        # --- Calculate NDCG ---
        if df_distances is not None:
            
            if query_col_spec:
                df_distances_filtered = df_distances[df_distances['column_1'].isin(query_cols)]
            else:
                df_distances_filtered = df_distances
                
            ndcg_val = self._calculate_ndcg(df_distances_filtered, gt_map, k)
            eval_metrics[f'ndcg@{k}'] = ndcg_val
            print(config.add_timestamp(config.color_text(f"NDCG@{k}: {ndcg_val:.4f}", "green")))

        count_metrics = {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

        print(config.add_timestamp(config.color_text(f"Evaluation Report: Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1_score:.4f}", "green")))
        print(config.add_timestamp(config.color_text(f"True Positives: {true_positives}, False Positives: {false_positives}, False Negatives: {false_negatives}", "green")))
        return eval_metrics, count_metrics


    def compute_distances_and_evaluation(self,
        embeddings: Union[list[dict], str],
        dist_threshold: float = config.distance_threshold,
        output_directory: str = config.global_output_directory,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Computes cosine similarity, euclidean distance, and ANNS distances between all pairs of column embeddings, 
        and exports the results as CSV files for Neo4j nodes and edges.

        Args:
            embeddings: A list of dictionaries, each containing the embedding for a column and its file name,
                        as well as the semantic annotation for the column (if available). Alternatively, a string
                        path to a JSON file containing the embeddings.
            dist_threshold (float): The maximum euclidean distance for two columns to be considered joinable.
            output_directory: The directory where the output CSV files will be saved.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The computed distances DataFrame, the nodes DataFrame,
                and the edges DataFrame.
        """
        if not embeddings:
            print(config.add_timestamp(config.color_text("No embeddings provided. Skipping distance calculation and export.", "yellow")))
            return None
            
        if isinstance(embeddings, str):
            if not os.path.exists(embeddings):
                print(config.add_timestamp(config.color_text(f"Error: embeddings JSON file not found at {embeddings}", "red")))
            with open(embeddings, "r") as f:
                embeddings = json.load(f)

        results = []
        for k in config.top_k_neighbors:
            for dist in config.distance_threshold:
                # --- Compute Distances ---
                df_distances = self._compute_distances(
                    embeddings=embeddings,
                    k_neighbors=k)
                config.save_csv(df_distances, f"distances_{k}_dist_{dist}.csv")

                # --- Extract Joinable Columns ---
                df_joinable = self._extract_joinable_columns(
                    df_distances=df_distances,
                    dist_threshold=dist,
                    dist_mode='anns')
                config.save_csv(df_joinable, f"joinable_columns_{k}_dist_{dist}.csv")

                # --- Generate Evaluation Report ---
                evaluation_metrics, count_metrics = self._evaluation_report(df_joinable, df_distances, k)
                results.append({
                    'distance_threshold': dist,
                    'k_neighbors': k,
                    'precision': evaluation_metrics['precision'],
                    'recall': evaluation_metrics['recall'],
                    'f1_score': evaluation_metrics['f1_score'],
                    'true_positives': count_metrics['true_positives'],
                    'false_positives': count_metrics['false_positives'],
                    'false_negatives': count_metrics['false_negatives']
                })

                # Add NDCG to results if computed
                if f'ndcg@{k}' in evaluation_metrics:
                    results[-1][f'ndcg@{k}'] = evaluation_metrics[f'ndcg@{k}']
        
        config.save_csv(pd.DataFrame(results), "evaluation_results.csv")
        return df_distances, df_joinable, evaluation_metrics