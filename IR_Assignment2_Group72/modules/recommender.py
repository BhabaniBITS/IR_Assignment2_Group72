"""
recommender.py
Recommendation module (Section E): content-based, collaborative and hybrid,
each returning Top-K recommendations with similarity/predicted scores.

Since the corpus has no real user click history, a synthetic but structured
interaction log is generated (users show category-biased preferences with
some noise) purely to demonstrate collaborative filtering mechanics. This is
disclosed clearly in the UI / report; the content-based path uses only the
document text itself and needs no synthetic data.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def content_based_recommend(index, seed_doc_id, top_k=5):
    pos = index.pos_of_doc_id(seed_doc_id)
    if pos is None:
        return []
    sims = cosine_similarity(index.tfidf_matrix[pos], index.tfidf_matrix).ravel()
    order = np.argsort(sims)[::-1]
    results = []
    for i in order:
        if index.doc_ids[i] == seed_doc_id:
            continue
        results.append((index.doc_ids[i], float(sims[i])))
        if len(results) >= top_k:
            break
    return results


def generate_synthetic_interactions(index, n_users=25, seed=42):
    """Create a synthetic implicit-feedback matrix: each user has a favourite
    category and rates documents in that category higher (with noise), so
    item-based collaborative filtering has real signal to recover."""
    rng = np.random.default_rng(seed)
    categories = sorted(set(index.categories))
    rows = []
    for u in range(n_users):
        fav_cat = rng.choice(categories)
        n_ratings = rng.integers(8, 20)
        candidate_positions = [i for i, c in enumerate(index.categories)]
        chosen = rng.choice(candidate_positions, size=min(n_ratings, len(candidate_positions)), replace=False)
        for pos in chosen:
            base = 4.2 if index.categories[pos] == fav_cat else 2.5
            rating = float(np.clip(rng.normal(base, 0.8), 1, 5))
            rows.append({"user_id": u, "doc_id": index.doc_ids[pos], "rating": round(rating, 2)})
    return pd.DataFrame(rows)


def collaborative_recommend(index, interactions_df, target_user_id, top_k=5):
    """Item-based collaborative filtering via cosine similarity of the
    user-item rating matrix."""
    if interactions_df is None or interactions_df.empty:
        return []
    pivot = interactions_df.pivot_table(index="user_id", columns="doc_id", values="rating", fill_value=0)
    if target_user_id not in pivot.index:
        return []

    item_sim = cosine_similarity(pivot.T.values)
    item_ids = list(pivot.columns)
    user_ratings = pivot.loc[target_user_id]
    rated_items = user_ratings[user_ratings > 0].index.tolist()

    scores = {}
    for item in item_ids:
        if item in rated_items:
            continue
        i_idx = item_ids.index(item)
        num, den = 0.0, 0.0
        for r_item in rated_items:
            r_idx = item_ids.index(r_item)
            sim = item_sim[i_idx, r_idx]
            num += sim * user_ratings[r_item]
            den += abs(sim)
        scores[item] = num / den if den > 0 else 0

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked


def hybrid_recommend(index, interactions_df, seed_doc_id, target_user_id, top_k=5, alpha=0.5):
    """Weighted hybrid of content-based and collaborative scores."""
    cb = dict(content_based_recommend(index, seed_doc_id, top_k=index.tfidf_matrix.shape[0]))
    cf = dict(collaborative_recommend(index, interactions_df, target_user_id, top_k=index.tfidf_matrix.shape[0]))

    def norm(d):
        if not d:
            return {}
        vals = np.array(list(d.values()))
        lo, hi = vals.min(), vals.max()
        if hi - lo < 1e-9:
            return {k: 0.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    cb_n, cf_n = norm(cb), norm(cf)
    all_docs = set(cb_n) | set(cf_n)
    blended = {d: alpha * cb_n.get(d, 0) + (1 - alpha) * cf_n.get(d, 0) for d in all_docs}
    ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked
