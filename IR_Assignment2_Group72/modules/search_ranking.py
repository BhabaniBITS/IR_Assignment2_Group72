"""
search_ranking.py
Web Searching module (Section D).

- Query processing: same clean/tokenize/normalize pipeline as documents.
- Retrieval / ranking models:
    1. Vector Space Model (TF-IDF cosine similarity)
    2. BM25 (Okapi) - a standard "query optimization" improvement over plain
       cosine similarity, favours term saturation & document-length norm.
    3. PageRank - computed over a document link graph. If the corpus has no
       real crawled hyperlinks (e.g. bundled dataset run), a similarity graph
       is built by linking each document to its top-k nearest neighbours
       (content-based proxy for citation/hyperlink structure) so the
       algorithm can still be demonstrated meaningfully.
    4. HITS (Hub/Authority scores) - computed on the same graph.
- Final ranking blends content relevance with the graph-based importance
  score, and the module exposes both rankings so the UI can show *why*
  ranking changes when importance is taken into account.
"""

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from . import preprocessing as prep


def process_query(query, method="stem"):
    tokens = prep.normalize_tokens(prep.tokenize(query), method=method)
    return " ".join(tokens)


def vsm_search(index, query, top_k=10):
    q_processed = process_query(query)
    q_vec = index.vectorizer.transform([q_processed])
    sims = cosine_similarity(q_vec, index.tfidf_matrix).ravel()
    order = np.argsort(sims)[::-1][:top_k]
    return [(index.doc_ids[i], float(sims[i])) for i in order if sims[i] > 0]


def bm25_search(index, query, top_k=10, k1=1.5, b=0.75):
    """Manual BM25 implementation over the index's processed texts."""
    docs = [t.split() for t in index.processed_texts]
    N = len(docs)
    avgdl = np.mean([len(d) for d in docs]) if docs else 1
    q_tokens = process_query(query).split()

    df = {}
    for term in set(q_tokens):
        df[term] = sum(1 for d in docs if term in d)

    scores = np.zeros(N)
    for i, d in enumerate(docs):
        dl = len(d)
        term_freqs = {}
        for t in d:
            term_freqs[t] = term_freqs.get(t, 0) + 1
        score = 0.0
        for term in q_tokens:
            if term not in df or df[term] == 0:
                continue
            idf = np.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))
            f = term_freqs.get(term, 0)
            denom = f + k1 * (1 - b + b * dl / avgdl)
            score += idf * (f * (k1 + 1)) / (denom if denom > 0 else 1)
        scores[i] = score

    order = np.argsort(scores)[::-1][:top_k]
    return [(index.doc_ids[i], float(scores[i])) for i in order if scores[i] > 0]


# --------------------------------------------------------------- graph rank --

def build_graph(index, links_df, top_k_similarity=5, sim_threshold=0.15):
    """
    Build a directed document graph.
    If real crawled hyperlinks exist in `links_df`, use them.
    Otherwise fall back to a content-similarity kNN graph so PageRank/HITS
    still have a meaningful structure to operate on.
    """
    G = nx.DiGraph()
    for doc_id in index.doc_ids:
        G.add_node(doc_id)

    used_real_links = False
    if links_df is not None and len(links_df) > 0:
        id_set = set(index.doc_ids)
        for _, row in links_df.iterrows():
            if row["src_doc"] in id_set and row["dst_doc"] in id_set:
                G.add_edge(row["src_doc"], row["dst_doc"])
                used_real_links = True

    if not used_real_links:
        sims = cosine_similarity(index.tfidf_matrix)
        n = sims.shape[0]
        for i in range(n):
            row = sims[i].copy()
            row[i] = -1
            top_idx = np.argsort(row)[::-1][:top_k_similarity]
            for j in top_idx:
                if sims[i, j] >= sim_threshold:
                    G.add_edge(index.doc_ids[i], index.doc_ids[j])

    return G, used_real_links


def compute_pagerank(G, alpha=0.85):
    if G.number_of_edges() == 0:
        return {n: 1 / max(1, G.number_of_nodes()) for n in G.nodes()}
    return nx.pagerank(G, alpha=alpha)


def compute_hits(G):
    if G.number_of_edges() == 0:
        n = max(1, G.number_of_nodes())
        eq = {node: 1 / n for node in G.nodes()}
        return eq, dict(eq)
    try:
        hubs, authorities = nx.hits(G, max_iter=500, normalized=True)
    except nx.PowerIterationFailedConvergence:
        hubs = {n: 1 / G.number_of_nodes() for n in G.nodes()}
        authorities = dict(hubs)
    return hubs, authorities


def blended_ranking(content_scores, importance_scores, alpha=0.7):
    """content_scores, importance_scores: dict doc_id -> score (both will be
    min-max normalized). alpha weights content relevance vs graph importance."""
    def norm(d):
        if not d:
            return {}
        vals = np.array(list(d.values()))
        lo, hi = vals.min(), vals.max()
        if hi - lo < 1e-9:
            return {k: 0.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    c = norm(content_scores)
    imp = norm({k: importance_scores.get(k, 0) for k in content_scores})
    blended = {k: alpha * c.get(k, 0) + (1 - alpha) * imp.get(k, 0) for k in content_scores}
    return dict(sorted(blended.items(), key=lambda x: x[1], reverse=True))
