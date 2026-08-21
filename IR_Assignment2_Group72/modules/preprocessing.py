"""
preprocessing.py
Text preprocessing, feature engineering and statistical analysis (Section C).

Provides:
  - clean_text / tokenize / stopword removal
  - two comparable normalization strategies: stemming (Porter) vs lemmatization
    (WordNet, with a light rule-based fallback if the wordnet corpus isn't
    available on this machine) -> used for a comparative-analysis table.
  - TF-IDF and Count feature extraction
  - keyword extraction (top TF-IDF terms per document)
  - document profiling (length stats, lexical diversity, top terms)
  - document classification (Multinomial Naive Bayes vs Logistic Regression
    trained on the labelled portion of the corpus)
"""

import re
import string
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import text as sk_text
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from nltk.stem import PorterStemmer

STOPWORDS = set(sk_text.ENGLISH_STOP_WORDS)
_stemmer = PorterStemmer()

# Try to use NLTK's WordNet lemmatizer; fall back to a tiny rule-based
# suffix-stripping lemmatizer if the corpus isn't downloaded on this machine.
try:
    from nltk.stem import WordNetLemmatizer
    _lemmatizer = WordNetLemmatizer()
    _lemmatizer.lemmatize("test")  # force a lookup to see if data is present
    _HAS_WORDNET = True
except Exception:
    _HAS_WORDNET = False

    class _FallbackLemmatizer:
        """Very light rule-based fallback used only if WordNet data is
        unavailable on this machine (e.g. no internet to nltk downloader).
        Not linguistically complete, but keeps the pipeline runnable."""
        _rules = [("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")]

        def lemmatize(self, word, pos="n"):
            for suf, rep in self._rules:
                if word.endswith(suf) and len(word) - len(suf) + len(rep) >= 3:
                    return word[: -len(suf)] + rep
            return word

    _lemmatizer = _FallbackLemmatizer()


def clean_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    return [t for t in clean_text(text).split() if len(t) > 2 and t not in STOPWORDS]


def normalize_tokens(tokens, method="stem"):
    if method == "stem":
        return [_stemmer.stem(t) for t in tokens]
    elif method == "lemma":
        return [_lemmatizer.lemmatize(t, pos="v") for t in tokens]
    return tokens


def preprocess_corpus(texts, method="stem"):
    """Return list of space-joined normalized-token strings, ready for vectorizing."""
    out = []
    for t in texts:
        toks = normalize_tokens(tokenize(t), method=method)
        out.append(" ".join(toks))
    return out


# ---------------------------------------------------------------- features --

def build_tfidf(corpus_processed, max_features=5000, ngram_range=(1, 1)):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vec.fit_transform(corpus_processed)
    return vec, X


def build_count(corpus_processed, max_features=5000):
    vec = CountVectorizer(max_features=max_features)
    X = vec.fit_transform(corpus_processed)
    return vec, X


def top_keywords_per_doc(vec: TfidfVectorizer, X, doc_idx, top_n=10):
    terms = np.array(vec.get_feature_names_out())
    row = X[doc_idx].toarray().ravel()
    top_ids = row.argsort()[::-1][:top_n]
    return [(terms[i], round(float(row[i]), 4)) for i in top_ids if row[i] > 0]


def corpus_keyword_frequency(corpus_processed, top_n=25):
    c = Counter()
    for doc in corpus_processed:
        c.update(doc.split())
    return c.most_common(top_n)


def document_profile(raw_text, processed_text):
    words = raw_text.split()
    unique_words = set(w.lower() for w in words)
    proc_tokens = processed_text.split()
    return {
        "char_count": len(raw_text),
        "word_count": len(words),
        "unique_word_count": len(unique_words),
        "lexical_diversity": round(len(unique_words) / max(1, len(words)), 3),
        "avg_word_length": round(np.mean([len(w) for w in words]) if words else 0, 2),
        "processed_token_count": len(proc_tokens),
    }


# ---------------------------------------------------------- classification --

def train_classifier(corpus_processed, labels, model="nb", test_size=0.25, seed=42):
    """Train a document classifier (used for auto-labelling crawled/unlabelled
    docs, and for the preprocessing comparative-analysis table)."""
    vec = TfidfVectorizer(max_features=5000)
    X = vec.fit_transform(corpus_processed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    clf = MultinomialNB() if model == "nb" else LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1_macro": round(f1_score(y_test, preds, average="macro"), 4),
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
    }
    return vec, clf, metrics


def compare_preprocessing_strategies(raw_texts, labels):
    """Comparative analysis of different preprocessing/feature-extraction
    strategies (Section C requirement): stem+TFIDF, lemma+TFIDF, stem+Count,
    raw(no-normalize)+TFIDF -- each scored via a classification proxy task."""
    results = []
    configs = [
        ("Stemming + TF-IDF", "stem", "tfidf"),
        ("Lemmatization + TF-IDF", "lemma", "tfidf"),
        ("Stemming + Count", "stem", "count"),
        ("No normalization + TF-IDF", "none", "tfidf"),
    ]
    for label, norm_method, feat_method in configs:
        processed = preprocess_corpus(raw_texts, method=norm_method)
        if feat_method == "tfidf":
            vec = TfidfVectorizer(max_features=5000)
        else:
            vec = CountVectorizer(max_features=5000)
        X = vec.fit_transform(processed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.25, random_state=42, stratify=labels
        )
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        results.append({
            "Strategy": label,
            "Vocabulary Size": X.shape[1],
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "F1 (macro)": round(f1_score(y_test, preds, average="macro"), 4),
        })
    return pd.DataFrame(results)
