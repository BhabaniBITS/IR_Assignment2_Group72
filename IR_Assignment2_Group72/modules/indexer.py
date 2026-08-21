"""
indexer.py
Index management: builds and maintains an in-memory inverted index plus the
TF-IDF document-term matrix used by search & ranking. The "index" is rebuilt
from the SQLite document store on demand (Index Management tab) so it always
reflects the latest crawl.
"""

from collections import defaultdict
import numpy as np

from . import preprocessing as prep


class IRIndex:
    def __init__(self):
        self.doc_ids = []
        self.raw_texts = []
        self.titles = []
        self.categories = []
        self.processed_texts = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.inverted_index = defaultdict(set)   # term -> set(doc positions)
        self.vocab_size = 0
        self.built = False

    def build(self, docs_df, norm_method="stem"):
        self.doc_ids = docs_df["doc_id"].tolist()
        self.raw_texts = docs_df["content"].fillna("").tolist()
        self.titles = docs_df["title"].fillna("").tolist()
        self.categories = docs_df["category"].fillna("unlabeled").tolist()

        self.processed_texts = prep.preprocess_corpus(self.raw_texts, method=norm_method)
        self.vectorizer, self.tfidf_matrix = prep.build_tfidf(self.processed_texts, max_features=8000)
        self.vocab_size = len(self.vectorizer.get_feature_names_out())

        self.inverted_index = defaultdict(set)
        for pos, text in enumerate(self.processed_texts):
            for term in set(text.split()):
                self.inverted_index[term].add(pos)

        self.built = True
        return self.stats()

    def stats(self):
        if not self.built:
            return {}
        doc_lengths = [len(t.split()) for t in self.processed_texts]
        return {
            "num_documents": len(self.doc_ids),
            "vocabulary_size": self.vocab_size,
            "avg_doc_length_tokens": round(np.mean(doc_lengths), 2) if doc_lengths else 0,
            "max_doc_length_tokens": max(doc_lengths) if doc_lengths else 0,
            "min_doc_length_tokens": min(doc_lengths) if doc_lengths else 0,
            "index_postings": len(self.inverted_index),
            "categories": sorted(set(self.categories)),
        }

    def postings(self, term):
        term = term.lower().strip()
        positions = self.inverted_index.get(term, set())
        return [self.doc_ids[p] for p in positions]

    def pos_of_doc_id(self, doc_id):
        try:
            return self.doc_ids.index(doc_id)
        except ValueError:
            return None
