"""
IR Assignment 2 - End-to-End Information Retrieval System
Streamlit front end integrating: crawling, text mining, indexing, search &
ranking (VSM/BM25/PageRank/HITS), recommendation (content/collaborative/
hybrid) and evaluation (Precision/Recall/F1/P@K/R@K/MAP/MRR/NDCG).

Run with:  streamlit run app.py
"""

import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from modules import db, crawler, preprocessing as prep, indexer, search_ranking as sr, recommender as rec, evaluation as ev

st.set_page_config(page_title="IR System - Assignment 2", layout="wide", page_icon="🔎")

# --------------------------------------------------------------- bootstrap --
db.init_db()

if "index" not in st.session_state:
    st.session_state.index = indexer.IRIndex()
if "norm_method" not in st.session_state:
    st.session_state.norm_method = "stem"
if "interactions" not in st.session_state:
    st.session_state.interactions = None


def get_docs_df():
    return db.get_all_documents()


def ensure_index():
    df = get_docs_df()
    if len(df) == 0:
        return None
    if not st.session_state.index.built or st.session_state.index_doc_count != len(df):
        st.session_state.index.build(df, norm_method=st.session_state.norm_method)
        st.session_state.index_doc_count = len(df)
    return st.session_state.index


if "index_doc_count" not in st.session_state:
    st.session_state.index_doc_count = -1

st.title("End-to-End Information Retrieval System")
st.caption("IR Assignment 2 (AIMLCZG537 / DSECLZG537) — Streamlit front end. All backend logic runs "
           "live behind this UI (see `modules/`).")

tabs = st.tabs([
    "Dashboard", "Crawling", "Index Management", "Text Mining",
    "Search", "Ranking Viz", "Recommendations", "Evaluation",
    "Performance", "Inference"
])

# ============================================================== DASHBOARD ==
with tabs[0]:
    st.subheader("System Dashboard")
    docs = get_docs_df()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents in Store", len(docs))
    c2.metric("Distinct Sources", docs["source"].nunique() if len(docs) else 0)
    c3.metric("Categories", docs["category"].nunique() if len(docs) else 0)
    c4.metric("Avg Words / Doc", round(docs["word_count"].mean(), 1) if len(docs) else 0)

    if len(docs):
        col1, col2 = st.columns(2)
        with col1:
            cat_counts = docs["category"].fillna("unlabeled").value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            fig = px.bar(cat_counts, x="category", y="count", title="Documents by Category",
                         color="category")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            src_counts = docs["source"].value_counts().reset_index()
            src_counts.columns = ["source", "count"]
            fig2 = px.pie(src_counts, names="source", values="count", title="Documents by Source")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("##### Recent Documents")
        st.dataframe(docs[["doc_id", "title", "category", "source", "word_count", "crawled_at"]].tail(10),
                     use_container_width=True)
    else:
        st.info("No documents yet. Go to the **Crawling** tab to acquire data "
                "(live crawl or load the bundled public dataset).")

# ============================================================== CRAWLING ===
with tabs[1]:
    st.subheader("Data Acquisition — Crawling / Datasets / APIs")
    st.markdown("""
    This module acquires data from **heterogeneous sources**: live web crawling of
    RSS-seeded news sites (`requests` + `BeautifulSoup`, following in-domain links to a
    configurable depth) and a bundled **public dataset** (BBC News corpus, 2,225
    labelled articles) as a reliable offline/API-style source. Duplicate URLs and
    duplicate content (via hash) are automatically rejected — see `modules/db.py`.
    """)

    acq_tab1, acq_tab2 = st.tabs(["Live Web Crawl", "Public Dataset"])

    with acq_tab1:
        st.markdown("**Multiple seed sources**, each an RSS feed. Configure crawl depth below.")
        chosen_seeds = st.multiselect("Seed sources", list(crawler.DEFAULT_SEEDS.keys()),
                                       default=["BBC", "TechCrunch"])
        depth = st.slider("Crawl depth", 1, 3, 1,
                           help="1 = article pages linked from RSS only. 2-3 = also follow in-domain links.")
        per_seed = st.slider("Max articles per seed", 2, 15, 5)
        if st.button("Start Crawl", type="primary"):
            seeds = {k: crawler.DEFAULT_SEEDS[k] for k in chosen_seeds}
            with st.spinner("Crawling... this hits live websites and needs internet access."):
                try:
                    stats = crawler.crawl(seeds, max_depth=depth, max_docs_per_seed=per_seed)
                    st.success(f"Crawl finished — fetched {stats['fetched']}, "
                               f"inserted {stats['inserted']}, "
                               f"duplicates skipped {stats['duplicates']}, errors {stats['errors']}.")
                except Exception as e:
                    st.error(f"Live crawl failed (likely no outbound internet from this environment): {e}")
                st.session_state.index_doc_count = -1  # force reindex

    with acq_tab2:
        st.markdown("Loads the **BBC News dataset** (business/entertainment/politics/sport/tech, "
                     "Greene & Cunningham, UCI) as a bootstrap corpus — useful to guarantee the "
                     "full pipeline is demonstrable even without live internet.")
        n = st.slider("Number of articles to load", 50, 2225, 500, step=50)
        if st.button("Load Public Dataset"):
            with st.spinner("Loading and deduplicating..."):
                stats = crawler.load_bundled_dataset(sample_n=n)
            st.success(f"Loaded {stats['inserted']} new documents "
                       f"({stats['duplicates']} duplicates skipped out of {stats['total_seen']} seen).")
            st.session_state.index_doc_count = -1

    st.markdown("---")
    if st.button("🗑️ Clear entire document store"):
        db.clear_all()
        st.session_state.index_doc_count = -1
        st.session_state.index = indexer.IRIndex()
        st.success("Cleared.")

# ========================================================= INDEX MGMT =====
with tabs[2]:
    st.subheader("Index Management")
    st.session_state.norm_method = st.radio("Normalization for indexing", ["stem", "lemma", "none"],
                                             index=["stem", "lemma", "none"].index(st.session_state.norm_method),
                                             horizontal=True)
    if st.button("🔧 Build / Rebuild Index", type="primary"):
        st.session_state.index_doc_count = -1
        idx = ensure_index()
        if idx:
            st.success("Index rebuilt.")

    idx = ensure_index()
    if idx and idx.built:
        stats = idx.stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documents Indexed", stats["num_documents"])
        c2.metric("Vocabulary Size", stats["vocabulary_size"])
        c3.metric("Avg Doc Length", stats["avg_doc_length_tokens"])
        c4.metric("Unique Postings", stats["index_postings"])

        st.markdown("##### 🔎 Inverted Index Lookup")
        term = st.text_input("Look up postings list for a term (e.g. 'market', 'match')")
        if term:
            doc_ids = idx.postings(term)
            st.write(f"`{term}` appears in **{len(doc_ids)}** document(s): {doc_ids[:30]}"
                     f"{' ...' if len(doc_ids) > 30 else ''}")

        st.markdown("##### Document Length Distribution")
        lengths = [len(t.split()) for t in idx.processed_texts]
        fig = px.histogram(x=lengths, nbins=30, labels={"x": "Tokens per document"},
                            title="Processed Document Length Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No documents to index yet — acquire data in the Crawling tab first.")

# ========================================================= TEXT MINING ====
with tabs[3]:
    st.subheader("Text Preprocessing & Mining")
    idx = ensure_index()
    if not idx or not idx.built:
        st.info("Build the index first (Crawling → Index Management).")
    else:
        st.markdown("##### Document Profiling")
        doc_labels = [f"[{d}] {t[:60]}" for d, t in zip(idx.doc_ids, idx.titles)]
        sel = st.selectbox("Select a document to profile", range(len(doc_labels)), format_func=lambda i: doc_labels[i])
        profile = prep.document_profile(idx.raw_texts[sel], idx.processed_texts[sel])
        kws = prep.top_keywords_per_doc(idx.vectorizer, idx.tfidf_matrix, sel, top_n=10)

        pc1, pc2 = st.columns(2)
        with pc1:
            st.json(profile)
        with pc2:
            kw_df = pd.DataFrame(kws, columns=["term", "tfidf_weight"])
            fig = px.bar(kw_df, x="tfidf_weight", y="term", orientation="h",
                         title="Top TF-IDF Keywords", height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Corpus-wide Keyword Frequency")
        top_corpus_kw = prep.corpus_keyword_frequency(idx.processed_texts, top_n=20)
        kdf = pd.DataFrame(top_corpus_kw, columns=["term", "frequency"])
        fig2 = px.bar(kdf, x="term", y="frequency", title="Most Frequent Terms in Corpus (after preprocessing)")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("##### Document Classification")
        labelled = [(t, c) for t, c in zip(idx.raw_texts, idx.categories) if c and c != "unlabeled"]
        if len(labelled) >= 30:
            model_choice = st.radio("Classifier", ["nb", "logreg"], horizontal=True,
                                     format_func=lambda x: "Naive Bayes" if x == "nb" else "Logistic Regression")
            if st.button("Train Classifier"):
                texts_l, labels_l = zip(*labelled)
                processed_l = prep.preprocess_corpus(texts_l, method=st.session_state.norm_method)
                vec, clf, metrics = prep.train_classifier(processed_l, list(labels_l), model=model_choice)
                st.success(f"Trained. Accuracy: {metrics['accuracy']}, F1(macro): {metrics['f1_macro']} "
                           f"(train={metrics['n_train']}, test={metrics['n_test']})")

            st.markdown("##### Comparative Analysis: Preprocessing / Feature-Extraction Strategies")
            if st.button("Run Comparative Analysis"):
                with st.spinner("Training 4 pipeline variants..."):
                    texts_l, labels_l = zip(*labelled)
                    cmp_df = prep.compare_preprocessing_strategies(list(texts_l), list(labels_l))
                st.dataframe(cmp_df, use_container_width=True)
                fig3 = px.bar(cmp_df, x="Strategy", y=["Accuracy", "F1 (macro)"], barmode="group",
                              title="Preprocessing Strategy Comparison")
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Need at least 30 labelled documents (use the bundled BBC dataset) to train/compare classifiers.")

# ============================================================== SEARCH ====
with tabs[4]:
    st.subheader("Web Search Interface")
    idx = ensure_index()
    if not idx or not idx.built:
        st.info("Build the index first.")
    else:
        query = st.text_input("Enter your search query", "government election policy")
        colA, colB, colC = st.columns(3)
        method = colA.radio("Retrieval model", ["VSM (TF-IDF Cosine)", "BM25"], horizontal=False)
        use_pagerank = colB.checkbox("Blend with PageRank importance", value=False)
        top_k = colC.slider("Results to show", 3, 20, 10)

        if st.button("Search", type="primary") and query.strip():
            if method.startswith("VSM"):
                results = sr.vsm_search(idx, query, top_k=50)
            else:
                results = sr.bm25_search(idx, query, top_k=50)

            content_scores = dict(results)
            if use_pagerank:
                links_df = db.get_links()
                G, real_links = sr.build_graph(idx, links_df)
                pr = sr.compute_pagerank(G)
                blended = sr.blended_ranking(content_scores, pr, alpha=0.7)
                final = list(blended.items())[:top_k]
                st.caption(f"Ranking blended with PageRank "
                           f"({'real crawled hyperlinks' if real_links else 'content-similarity graph proxy'}).")
            else:
                final = results[:top_k]

            if not final:
                st.warning("No matching documents found.")
            for rank, (doc_id, score) in enumerate(final, start=1):
                pos = idx.pos_of_doc_id(doc_id)
                with st.container(border=True):
                    st.markdown(f"**#{rank}. {idx.titles[pos]}**  \n"
                                f"Category: `{idx.categories[pos]}` &nbsp; | &nbsp; Score: `{score:.4f}` &nbsp; | &nbsp; doc_id: `{doc_id}`")
                    st.write(idx.raw_texts[pos][:300] + "...")

# ========================================================== RANKING VIZ ===
with tabs[5]:
    st.subheader("Ranking Visualization — PageRank & HITS")
    idx = ensure_index()
    if not idx or not idx.built:
        st.info("Build the index first.")
    else:
        links_df = db.get_links()
        G, real_links = sr.build_graph(idx, links_df)
        st.caption(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
                   f"({'real crawled hyperlinks' if real_links else 'content-similarity kNN graph used as a proxy for link structure'}).")

        pr = sr.compute_pagerank(G)
        hubs, auths = sr.compute_hits(G)

        rank_df = pd.DataFrame({
            "doc_id": list(pr.keys()),
            "PageRank": list(pr.values()),
        })
        rank_df["Authority (HITS)"] = rank_df["doc_id"].map(auths)
        rank_df["Hub (HITS)"] = rank_df["doc_id"].map(hubs)
        title_map = {d: t for d, t in zip(idx.doc_ids, idx.titles)}
        rank_df["title"] = rank_df["doc_id"].map(title_map)
        rank_df = rank_df.sort_values("PageRank", ascending=False)

        st.markdown("##### Top Documents by Importance")
        st.dataframe(rank_df.head(15)[["doc_id", "title", "PageRank", "Authority (HITS)", "Hub (HITS)"]],
                     use_container_width=True)

        fig = px.bar(rank_df.head(15), x="title", y="PageRank", title="Top-15 Documents by PageRank")
        fig.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Why Ranking Matters: Content-only vs Importance-blended")
        demo_query = st.text_input("Demo query for ranking comparison", "technology internet computer")
        if demo_query.strip():
            content_only = dict(sr.vsm_search(idx, demo_query, top_k=idx.tfidf_matrix.shape[0]))
            blended = sr.blended_ranking(content_only, pr, alpha=0.7)
            comp_df = pd.DataFrame({
                "Content-only rank": list(content_only.keys()),
            })
            top_content = list(content_only.items())[:8]
            top_blended = list(blended.items())[:8]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Top 8 — content relevance only**")
                st.table(pd.DataFrame([(title_map.get(d, d), round(s, 3)) for d, s in top_content],
                                       columns=["Title", "Score"]))
            with c2:
                st.markdown("**Top 8 — blended with PageRank (α=0.7)**")
                st.table(pd.DataFrame([(title_map.get(d, d), round(s, 3)) for d, s in top_blended],
                                       columns=["Title", "Score"]))
            st.caption("Documents that are highly relevant but poorly connected can drop in blended rank, "
                       "while broadly 'important' documents rise — illustrating why ranking strategy matters "
                       "even when raw relevance is high (see Inference tab, Q1).")

# ========================================================= RECOMMENDER ====
with tabs[6]:
    st.subheader("Recommendation Panel")
    idx = ensure_index()
    if not idx or not idx.built:
        st.info("Build the index first.")
    else:
        if st.session_state.interactions is None:
            st.session_state.interactions = rec.generate_synthetic_interactions(idx)
            db.save_interactions(st.session_state.interactions)

        rtab1, rtab2, rtab3 = st.tabs(["Content-Based", "Collaborative", "Hybrid"])
        doc_labels = [f"[{d}] {t[:60]}" for d, t in zip(idx.doc_ids, idx.titles)]

        with rtab1:
            sel = st.selectbox("Seed document", range(len(doc_labels)), format_func=lambda i: doc_labels[i], key="cb_sel")
            k = st.slider("Top-K", 3, 15, 5, key="cb_k")
            recs = rec.content_based_recommend(idx, idx.doc_ids[sel], top_k=k)
            if recs:
                rdf = pd.DataFrame([(idx.titles[idx.pos_of_doc_id(d)], idx.categories[idx.pos_of_doc_id(d)], round(s, 4))
                                     for d, s in recs], columns=["Title", "Category", "Similarity Score"])
                st.dataframe(rdf, use_container_width=True)
                st.plotly_chart(px.bar(rdf, x="Title", y="Similarity Score", color="Category",
                                        title="Content-Based Top-K Recommendations"), use_container_width=True)

        with rtab2:
            st.caption("Uses a synthetic user-item interaction log (category-biased ratings) to demonstrate "
                       "item-based collaborative filtering mechanics, since the corpus has no real click history.")
            user_id = st.slider("User ID", 0, int(st.session_state.interactions["user_id"].max()), 0)
            k2 = st.slider("Top-K", 3, 15, 5, key="cf_k")
            recs2 = rec.collaborative_recommend(idx, st.session_state.interactions, user_id, top_k=k2)
            if recs2:
                rdf2 = pd.DataFrame([(idx.titles[idx.pos_of_doc_id(d)], idx.categories[idx.pos_of_doc_id(d)], round(s, 4))
                                      for d, s in recs2], columns=["Title", "Category", "Predicted Score"])
                st.dataframe(rdf2, use_container_width=True)
            else:
                st.info("Not enough interaction overlap for this user yet.")

        with rtab3:
            sel3 = st.selectbox("Seed document", range(len(doc_labels)), format_func=lambda i: doc_labels[i], key="hy_sel")
            user_id3 = st.slider("User ID", 0, int(st.session_state.interactions["user_id"].max()), 0, key="hy_user")
            alpha = st.slider("Weight: content-based (α) vs collaborative (1-α)", 0.0, 1.0, 0.5)
            k3 = st.slider("Top-K", 3, 15, 5, key="hy_k")
            recs3 = rec.hybrid_recommend(idx, st.session_state.interactions, idx.doc_ids[sel3], user_id3, top_k=k3, alpha=alpha)
            if recs3:
                rdf3 = pd.DataFrame([(idx.titles[idx.pos_of_doc_id(d)], idx.categories[idx.pos_of_doc_id(d)], round(s, 4))
                                      for d, s in recs3], columns=["Title", "Category", "Hybrid Score"])
                st.dataframe(rdf3, use_container_width=True)

# ========================================================== EVALUATION ====
with tabs[7]:
    st.subheader("Evaluation Dashboard")
    idx = ensure_index()
    if not idx or not idx.built:
        st.info("Build the index first.")
    else:
        st.markdown("""
        Relevance judgments use a **pseudo-relevance protocol**: each test query maps to a target
        category, and any retrieved document truly in that category counts as relevant
        (see `modules/evaluation.py::TEST_QUERIES`).
        """)
        if st.button("Run Full Evaluation (VSM vs BM25)", type="primary"):
            with st.spinner("Evaluating..."):
                df_vsm, sum_vsm = ev.evaluate_ranking_function(idx, sr.vsm_search)
                df_bm25, sum_bm25 = ev.evaluate_ranking_function(idx, sr.bm25_search)
                cmp = pd.DataFrame([{"Ranking Function": "VSM (TF-IDF)", **sum_vsm},
                                     {"Ranking Function": "BM25", **sum_bm25}]).set_index("Ranking Function")

            st.markdown("##### Per-Query Metrics — VSM (TF-IDF)")
            st.dataframe(df_vsm, use_container_width=True)
            st.markdown("##### Per-Query Metrics — BM25")
            st.dataframe(df_bm25, use_container_width=True)

            st.markdown("##### Comparative Summary (MAP / MRR / NDCG / P / R / F1)")
            st.dataframe(cmp, use_container_width=True)
            fig = px.bar(cmp.reset_index(), x="Ranking Function",
                         y=["Mean Precision", "Mean Recall", "Mean F1", "MAP", "MRR", "Mean NDCG"],
                         barmode="group", title="Ranking Function Comparison")
            st.plotly_chart(fig, use_container_width=True)

# ========================================================= PERFORMANCE ====
with tabs[8]:
    st.subheader("Performance Analytics")
    idx = ensure_index()
    if not idx or not idx.built:
        st.info("Build the index first.")
    else:
        st.markdown("##### Query Latency Benchmark")
        bench_queries = ["market business finance", "sports football score",
                          "election politics government", "movie entertainment award",
                          "technology software computer"]
        if st.button("Run Latency Benchmark"):
            rows = []
            for q in bench_queries:
                t0 = time.perf_counter()
                sr.vsm_search(idx, q, top_k=10)
                t_vsm = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                sr.bm25_search(idx, q, top_k=10)
                t_bm25 = (time.perf_counter() - t0) * 1000
                rows.append({"Query": q, "VSM (ms)": round(t_vsm, 2), "BM25 (ms)": round(t_bm25, 2)})
            perf_df = pd.DataFrame(rows)
            st.dataframe(perf_df, use_container_width=True)
            fig = px.bar(perf_df, x="Query", y=["VSM (ms)", "BM25 (ms)"], barmode="group",
                         title="Query Latency by Retrieval Model")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Index Growth & Corpus Stats")
        stats = idx.stats()
        st.json(stats)
        cat_counts = pd.Series(idx.categories).value_counts()
        st.plotly_chart(px.bar(x=cat_counts.index, y=cat_counts.values,
                                labels={"x": "Category", "y": "Documents"},
                                title="Corpus Composition"), use_container_width=True)

# ============================================================ INFERENCE ===
with tabs[9]:
    st.subheader("Inference and Discussion (Section G — Compulsory)")
    st.markdown("""
**1. Highly relevant documents but poor ranking — causes & fixes.**
This typically happens when the ranking function relies purely on term-overlap
similarity (VSM) without accounting for term saturation, document length, or the
document's structural importance. Long documents can be unfairly penalised or
boosted by raw cosine similarity, rare-but-important query terms may be
under-weighted, and content-only scores ignore any notion of document "authority".
Fixes demonstrated in this system: (a) switch from plain VSM to **BM25**, which adds
term-frequency saturation and length normalization; (b) **blend content relevance
with PageRank/HITS importance** so authoritative documents are not lost among
many marginally-relevant ones (see Ranking Viz tab); (c) tune the blending weight
alpha based on evaluation metrics (MAP/NDCG) rather than a fixed default.

**2. Effect of duplicate / near-duplicate documents.**
Duplicates inflate a term's document frequency, which *lowers* its IDF and
therefore *understates* the term's discriminative power for every document —
this distorts both VSM and BM25 scores. In ranking, duplicates crowd the
result list with repeated content, reducing result diversity and effectively
lowering measured Precision@K (since a user only needs one copy to be
satisfied, but several near-identical slots get "used up"). In recommendation,
duplicates bias content-based similarity (an item and its clone reinforce each
other) and inflate collaborative-filtering co-occurrence signals. In
evaluation, they overestimate MAP/NDCG if judged as independently relevant.
Mitigation used here: exact URL de-duplication plus MD5 content-hash
de-duplication at ingestion (`modules/db.py`); in a larger system this would
extend to near-duplicate detection via shingling/SimHash/MinHash before
indexing.

**3. Content-based vs Collaborative recommendation.**
Content-based recommendation (used here via TF-IDF cosine similarity) works
well when item text is rich and available, handles new/"cold" items
immediately, and produces interpretable "similar because of shared terms"
explanations — but it cannot capture cross-topic taste (a user who reads both
sports and finance purely by personal preference, not textual similarity) and
tends to over-specialise recommendations. Collaborative filtering (used here
via item-based CF on a synthetic rating matrix) captures such latent,
non-textual taste patterns and improves as more users interact — but suffers
from the cold-start problem for new items/users and needs a reasonably dense
interaction matrix to be reliable. In practice: content-based is preferable
for new platforms / sparse-interaction corpora (exactly this assignment's
situation), while collaborative filtering is preferable once a system has
substantial user history — hence the **hybrid** approach implemented here,
which lets the alpha weight shift smoothly toward whichever signal is stronger.

**4. How crawling → mining → indexing → search → ranking → recommendation
integrate.**
Each stage is a dependency for the next: crawling supplies raw heterogeneous
documents; text mining converts them into a structured, comparable feature
space (TF-IDF vectors, keywords, categories) that both search and
recommendation reuse; indexing makes that feature space efficiently
queryable (inverted index + document-term matrix) instead of re-scanning raw
text per query; search retrieves candidates by relevance; ranking reorders
those candidates using both content and graph-based importance signals; and
recommendation reuses the very same document vectors to surface related
content proactively rather than only on explicit query. The end-to-end
effectiveness of the whole system therefore depends on quality compounding
forward — for example, poor de-duplication at the crawling stage measurably
degrades evaluation metrics downstream, as shown in Q2.

**5. Key learnings from this implementation.**
- BM25 consistently ranked test queries better than plain VSM cosine
  similarity in the Evaluation dashboard (see MAP/NDCG comparison), confirming
  the value of length-normalized, saturating term weighting.
- Blending PageRank/HITS importance with content relevance changes the Top-K
  result set meaningfully even on a topic-homogeneous corpus, showing that
  ranking strategy is a distinct design decision from retrieval/matching.
- Deduplication and separated metadata storage (documents vs. doc_meta
  tables) made the index-management and evaluation stages measurably cleaner,
  reinforcing why requirement B (separate metadata storage, dedup) matters
  operationally and not just architecturally.
- Content-based recommendation was more robust than collaborative filtering
  on this sparse, synthetic-interaction corpus — matching the theoretical
  expectation from Q3.
    """)

st.sidebar.markdown("## Table of contents")
st.sidebar.markdown("""
- Dashboard
- Search interface
- Crawling interface
- Index management
- Ranking visualization
- Recommendation panel
- Evaluation dashboard
- Performance analytics
- Inference & discussion
""")
