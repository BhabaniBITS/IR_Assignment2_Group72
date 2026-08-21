"""
evaluation.py
Evaluation module (Section F): Precision, Recall, F1, P@K, R@K, MAP, MRR, NDCG.

Ground truth / relevance judgments:
Since the corpus is not hand-annotated with query-level relevance, we use a
standard, transparent pseudo-relevance protocol that is disclosed in the
report: a small bank of test queries is mapped to a target category (e.g.
query "stock market earnings" -> category "business"), and any retrieved
document that truly belongs to that category is treated as relevant. This
lets every IR metric below be computed exactly, and is a common practical
technique when no manual judgments are available for a course assignment.
"""

import numpy as np
import pandas as pd

TEST_QUERIES = {
    "stock market earnings profit": "business",
    "football match tournament score": "sport",
    "movie film actor award": "entertainment",
    "election government minister policy": "politics",
    "software computer technology internet": "tech",
}


def _relevance_vector(ranked_doc_ids, index, relevant_category):
    cat_by_doc = {d: c for d, c in zip(index.doc_ids, index.categories)}
    return [1 if cat_by_doc.get(d) == relevant_category else 0 for d in ranked_doc_ids]


def precision_recall_f1(rel_vec, total_relevant_in_corpus):
    retrieved = len(rel_vec)
    tp = sum(rel_vec)
    precision = tp / retrieved if retrieved else 0
    recall = tp / total_relevant_in_corpus if total_relevant_in_corpus else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1


def precision_at_k(rel_vec, k):
    rel_vec_k = rel_vec[:k]
    return sum(rel_vec_k) / k if k else 0


def recall_at_k(rel_vec, k, total_relevant):
    rel_vec_k = rel_vec[:k]
    return sum(rel_vec_k) / total_relevant if total_relevant else 0


def average_precision(rel_vec):
    hits, sum_prec = 0, 0.0
    for i, rel in enumerate(rel_vec, start=1):
        if rel:
            hits += 1
            sum_prec += hits / i
    return sum_prec / hits if hits else 0


def reciprocal_rank(rel_vec):
    for i, rel in enumerate(rel_vec, start=1):
        if rel:
            return 1 / i
    return 0


def ndcg(rel_vec, k=None):
    if k:
        rel_vec = rel_vec[:k]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(rel_vec))
    ideal = sorted(rel_vec, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0


def evaluate_ranking_function(index, search_fn, k_values=(5, 10), top_k=20, **search_kwargs):
    """
    search_fn(index, query, top_k=...) -> list[(doc_id, score)]
    Runs every TEST_QUERIES query through search_fn and averages metrics.
    """
    rows = []
    for query, category in TEST_QUERIES.items():
        total_relevant = sum(1 for c in index.categories if c == category)
        if total_relevant == 0:
            continue
        results = search_fn(index, query, top_k=top_k, **search_kwargs)
        ranked_ids = [d for d, _ in results]
        rel_vec = _relevance_vector(ranked_ids, index, category)

        p, r, f1 = precision_recall_f1(rel_vec, total_relevant)
        row = {
            "Query": query,
            "Target Category": category,
            "Precision": round(p, 3),
            "Recall": round(r, 3),
            "F1": round(f1, 3),
            "AP": round(average_precision(rel_vec), 3),
            "RR": round(reciprocal_rank(rel_vec), 3),
            "NDCG": round(ndcg(rel_vec), 3),
        }
        for k in k_values:
            row[f"P@{k}"] = round(precision_at_k(rel_vec, k), 3)
            row[f"R@{k}"] = round(recall_at_k(rel_vec, k, total_relevant), 3)
            row[f"NDCG@{k}"] = round(ndcg(rel_vec, k), 3)
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = {
        "MAP": round(df["AP"].mean(), 4) if len(df) else 0,
        "MRR": round(df["RR"].mean(), 4) if len(df) else 0,
        "Mean NDCG": round(df["NDCG"].mean(), 4) if len(df) else 0,
        "Mean Precision": round(df["Precision"].mean(), 4) if len(df) else 0,
        "Mean Recall": round(df["Recall"].mean(), 4) if len(df) else 0,
        "Mean F1": round(df["F1"].mean(), 4) if len(df) else 0,
    }
    return df, summary


def compare_ranking_functions(index, functions: dict, **kwargs):
    """functions: {"VSM (TF-IDF)": vsm_search, "BM25": bm25_search, ...}
    Returns a comparison DataFrame of summary metrics per ranking function."""
    rows = []
    for name, fn in functions.items():
        _, summary = evaluate_ranking_function(index, fn, **kwargs)
        summary["Ranking Function"] = name
        rows.append(summary)
    df = pd.DataFrame(rows).set_index("Ranking Function")
    return df
