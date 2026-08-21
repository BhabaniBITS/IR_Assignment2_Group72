# End-to-End Information Retrieval System — News Domain
IR Assignment 2 (AIMLCZG537 / DSECLZG537, S2-25)

A single Streamlit application implementing the complete IR lifecycle over a
news-article corpus: crawling, text mining, indexing, search & ranking,
recommendation, and evaluation.

## Domain / Use case
News articles (business, entertainment, politics, sport, tech). Data is
acquired from two heterogeneous sources:
1. **Live web crawling** — RSS-seeded (BBC, TechCrunch, NDTV, ESPN, Reuters),
   followed by `requests` + `BeautifulSoup` article-page extraction with a
   configurable crawl depth (follows in-domain links from each article page).
2. **Public dataset** — the BBC News corpus (2,225 labelled articles;
   Greene & Cunningham, UCI), bundled as `data/bbc_dataset_clean.csv`, used
   as a reliable offline/API-style source so the system is fully demoable
   even without live internet access (e.g. inside a locked-down lab VM).

> Note: live crawling requires outbound internet access to the news
> websites. If the lab environment blocks outbound web traffic, use the
> **Public Dataset** tab in the Crawling page instead — every other module
> (mining, indexing, search, ranking, recommendation, evaluation) works
> identically regardless of data source.

## Installation

```bash
pip install -r requirements.txt
```

(Optional, only needed for the WordNet-based lemmatizer variant; the app
automatically falls back to a lightweight rule-based lemmatizer if this
isn't run / no internet is available):
```bash
python -c "import nltk; nltk.download('wordnet')"
```

## Running the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## Suggested workflow for a demo / screenshots
1. **Crawling tab** → Public Dataset sub-tab → load ~300-500 articles (or
   try Live Web Crawl if internet is available).
2. **Index Management** → Build/Rebuild Index.
3. **Text Mining** → inspect document profiles, keyword extraction, train
   the classifier, run the preprocessing comparative analysis.
4. **Search** → try VSM vs BM25, toggle PageRank blending.
5. **Ranking Viz** → inspect PageRank/HITS scores and the content-only vs
   blended ranking comparison.
6. **Recommendations** → content-based, collaborative, hybrid tabs.
7. **Evaluation** → run the full VSM vs BM25 evaluation (Precision, Recall,
   F1, P@K, R@K, MAP, MRR, NDCG).
8. **Performance** → latency benchmark + corpus stats.
9. **Inference** → the compulsory Section G write-up, ready to copy into the
   report.

## Project structure
```
app.py                     Streamlit front end (all 10 tabs)
modules/
  db.py                    SQLite storage — documents & metadata in
                            SEPARATE tables, plus link graph & interactions
  crawler.py                Heterogeneous acquisition: RSS+crawl / dataset
  preprocessing.py          Cleaning, stemming/lemmatization, TF-IDF,
                            keyword extraction, document profiling,
                            classification, comparative analysis
  indexer.py                Inverted index + TF-IDF matrix, index stats
  search_ranking.py         VSM, BM25, PageRank, HITS, blended ranking
  recommender.py            Content-based, collaborative, hybrid
  evaluation.py             Precision/Recall/F1/P@K/R@K/MAP/MRR/NDCG
data/
  bbc_dataset_clean.csv     Bundled public dataset
  ir_system.db              Created at runtime (SQLite)
requirements.txt
README.md
```

## Design notes relevant to the rubric
- **Duplicate handling**: exact URL uniqueness constraint + MD5 content-hash
  de-duplication at ingestion (`db.insert_document`).
- **Metadata stored separately from content**: `documents` table (content)
  vs. `doc_meta` table (word/char counts, hashes, fetch status) — a
  deliberate separation per the assignment's explicit requirement.
- **Configurable crawl depth & multiple seeds**: exposed directly in the
  Crawling tab UI (`crawler.crawl(seed_sources, max_depth, ...)`).
- **Ranking algorithm (PageRank/HITS)**: built over real crawled hyperlinks
  when available, else a content-similarity kNN graph is used as a
  transparent proxy so the algorithm is still meaningfully demonstrable on
  the bundled dataset (disclosed in-app).
- **Evaluation ground truth**: pseudo-relevance via query→category mapping,
  documented in `evaluation.py` and disclosed in the Evaluation tab.
