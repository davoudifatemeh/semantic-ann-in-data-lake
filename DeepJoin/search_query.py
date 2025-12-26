# DeepJoin Pipeline (Post-Training):
# 1. Load fine-tuned model
# 2. Build ANN index from test repository columns
# 3. Evaluate on queries (Precision, Recall, F1, NDCG)

import os
import torch
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from repositoryLoader import RepositoryLoader, column_to_text
from annIndex import ANNIndex
from math import log2
from config import MODEL_OUTPUT_PATH, INDEX_PATH, TEST_REPO, TOP_K, QUERY_FILE

def normalize_sentence(text: str) -> str:
    return " ".join(text.strip().lower().split())

def load_repository_sentences(pairs_path: str) -> list[str]:
    repo_sentences = []
    seen = set()

    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line)
            # print(pair)
            if len(pair) >= 2:
                sent = normalize_sentence(pair[1])
                if sent not in seen:
                    seen.add(sent)
                    repo_sentences.append(sent)
                # repo_sentences.append(pair[1])  # 2nd element = repository col
    print(f"Loaded {len(repo_sentences)} unique repository sentences from {pairs_path}")
    return repo_sentences

def load_query_sentences(path: str) -> list[str]:
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            val = json.loads(line)
            sent = val[0] if isinstance(val, list) else val
            queries.append(normalize_sentence(sent))
    print(f"Loaded {len(queries)} normalized query sentences from {path}")
    return queries


def build_query_ground_truth(pairs_path: str) -> dict[str, set[str]]:
    # Build ground truth mapping from test pairs JSONL file:
    # query_sentence -> set of joinable candidate sentences.

    ground_truth = {}

    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line)
            if len(pair) >= 2:
                if pair[0] is None or pair[1] is None:
                    continue
                query = normalize_sentence(pair[0])
                candidate = normalize_sentence(pair[1])
                if query not in ground_truth:
                    ground_truth[query] = set()
                ground_truth[query].add(candidate)

    print(f"Built query ground truth for {len(ground_truth)} unique query sentences "
          f"from {pairs_path}")
    return ground_truth

def compute_metrics(results, ground_truth, k=TOP_K):
    # Compute Precision, Recall, F1, NDCG, TP, FP, FN.
    eps = 1e-8
    tp = fp = fn = 0
    precisions, recalls, f1s, ndcgs = [], [], [], []

    for res in results:
        query = res["query"]
        retrieved = [cand for cand, _ in res["neighbors"]]
        true = ground_truth.get(query, set())

        retrieved_k = set(retrieved[:k])
        true_pos = len(retrieved_k & true)
        false_pos = len(retrieved_k - true)
        false_neg = len(true - retrieved_k)

        tp += true_pos
        fp += false_pos
        fn += false_neg

        # per-query metrics
        precision = true_pos / (len(retrieved_k) + eps) 
        recall = true_pos / (len(true) + eps)
        f1 = 2 * precision * recall / ((precision + recall) + eps)

        # NDCG
        dcg = sum(1 / log2(i + 2) for i, cand in enumerate(retrieved[:k]) if cand in true)
        idcg = sum(1 / log2(i + 2) for i in range(min(len(true), k)))
        ndcg = dcg / idcg if idcg > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        ndcgs.append(ndcg)
    
    precision_micro = tp / (tp + fp + eps)
    recall_micro = tp / (tp + fn + eps)
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + eps)
    metrics = {
        f"Precision@{k}": round(precision_micro, 4),
        f"Recall@{k}": round(recall_micro, 4),
        f"F1@{k}": round(f1_micro, 4),
        f"NDCG@{k}": round(np.mean(ndcgs), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

    return metrics
