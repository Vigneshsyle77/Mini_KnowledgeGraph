# modules/app.py
"""
AI-KnowMap - Streamlit app WITHOUT spaCy (cloud-friendly)
Place this file at Mini_KnowledgeGraph/modules/app.py
Run: streamlit run modules/app.py
"""

import os
import sys
import tempfile
import re
from pathlib import Path
import traceback
import pandas as pd
import streamlit as st

# ------------------------------------
# GLOBAL HELPER — FIXED
# Must be before pages use it
# ------------------------------------
def find_col(df, name_list):
    """
    Auto-detect a column whose lowercase name matches items in name_list.
    Example: find_col(df, ["subject"]) returns the actual column name.
    """
    if df is None or df.empty:
        return None
    
    lower_targets = [x.lower().strip() for x in name_list]

    for col in df.columns:
        if col.lower().strip() in lower_targets:
            return col
    return None

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional graph cleaner (if present)
try:
    from modules.graph_cleaner import clean_graph
except Exception:
    try:
        from graph_cleaner import clean_graph  # fallback when running inside modules/
    except Exception:
        clean_graph = None

# Visualization & ML
import networkx as nx
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Paths
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
UI_DIR = ROOT / "ui"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
UI_DIR.mkdir(parents=True, exist_ok=True)

ENTITIES_OUT = PROCESSED_DIR / "entities_out.csv"
RELATIONS_OUT = PROCESSED_DIR / "relations_out.csv"
GRAPH_HTML = UI_DIR / "knowledge_graph.html"

# -------------------------
# Lightweight heuristic NER (no spaCy)
# -------------------------
# patterns and heuristics
ORG_KEYWORDS = [
    "University", "Inc", "Corp", "Corporation", "Company", "Google", "Amazon", "NASA",
    "Microsoft", "Apple", "Services", "Institute", "Bank", "Harvard", "Stanford", "John Hopkins",
    "Goldman Sachs"
]
PERSON_TITLE_PATTERN = r'\b(Dr|Mr|Ms|Mrs|Prof)\.?\s+[A-Z][a-z]+\s?[A-Z]?[a-z]*\b'
YEAR_PATTERN = r'\b(19|20)\d{2}\b'
MONTH_PATTERN = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b'
PERCENT_PATTERN = r'\b\d+%'

def extract_entities_heuristic(text):
    """
    Returns list of (entity_text, label) using simple heuristics:
    - PERSON: title + name or two Capitalized words
    - ORG: contains org keywords or all-caps tokens (NASA)
    - DATE: year or month-year patterns
    - PERCENT: percent tokens
    """
    ents = []
    if not isinstance(text, str) or text.strip() == "":
        return ents

    # Dates (years)
    for m in re.finditer(YEAR_PATTERN, text):
        ents.append((m.group(0), "DATE"))

    # Months (e.g., June 10, 2024)
    for m in re.finditer(MONTH_PATTERN + r'(?:\s+\d{1,2},\s*\d{4})?', text):
        ents.append((m.group(0), "DATE"))

    # Percent
    for m in re.finditer(PERCENT_PATTERN, text):
        ents.append((m.group(0), "PERCENT"))

    # Person via titles
    for m in re.finditer(PERSON_TITLE_PATTERN, text):
        ents.append((m.group(0), "PERSON"))

    # Person via two capitalized words (good heuristic for names)
    # avoid matching sentence starts followed by month/word; check for two consecutive Titlecase words
    two_caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
    for t in two_caps:
        # skip if t contains organization keywords
        if any(k.lower() in t.lower() for k in ORG_KEYWORDS):
            continue
        # avoid catching long phrases (limit words)
        if len(t.split()) <= 3:
            ents.append((t, "PERSON"))

    # Orgs by keywords or all-caps tokens
    # keyword match
    for k in ORG_KEYWORDS:
        if re.search(r'\b' + re.escape(k) + r'\b', text):
            # extract exact token occurrences
            for m in re.finditer(r'\b' + re.escape(k) + r'\b', text):
                ents.append((m.group(0), "ORG"))
    # All-caps acronyms (NASA, AWS)
    for m in re.finditer(r'\b([A-Z]{2,10})\b', text):
        token = m.group(1)
        # skip short common words (like 'US' maybe ok)
        if token.isupper() and len(token) >= 2 and len(token) <= 10:
            ents.append((token, "ORG"))

    # deduplicate preserving order
    seen = set()
    filtered = []
    for e, label in ents:
        key = (e.strip(), label)
        if key not in seen and e.strip() != "":
            seen.add(key)
            filtered.append((e.strip(), label))
    return filtered

# -------------------------
# Lightweight relation extraction (verb keywords)
# -------------------------
REL_KEYWORDS = [
    "launched", "reported", "conducted", "collaborated", "partnered",
    "released", "held", "invested", "announced", "founded", "created"
]

def extract_triples_heuristic(text):
    """
    Return list of triples found in a single sentence text:
    - Search for REL_KEYWORDS; take subject as text before verb (first noun-ish token), object after verb (next noun phrase)
    - fallback: use first two heuristic entities if any
    """
    triples = []
    if not isinstance(text, str):
        return triples
    txt = text.strip()
    low = txt.lower()
    for rel_kw in REL_KEYWORDS:
        if rel_kw in low:
            # find index of verb
            idx = low.find(rel_kw)
            # subject candidate: chunk before verb (split by comma or 'with' or 'by' etc)
            before = txt[:idx].strip()
            after = txt[idx + len(rel_kw):].strip()
            # take last capitalized phrase before (heuristic)
            subj = None
            # often subject is first capitalized token(s) in 'before'
            match_subj = re.findall(r'([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)*)', before)
            if match_subj:
                subj = match_subj[-1]
            else:
                # fallback: first word of before
                subj = before.split()[-1] if before.split() else before

            # object heuristic: first capitalized phrase in after or first token
            match_obj = re.findall(r'([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)*)', after)
            if match_obj:
                obj = match_obj[0]
            else:
                # try first noun-like token after verb (simple)
                obj = after.split()[0] if after.split() else after
            triples.append((subj.strip(), rel_kw, obj.strip()))
    # fallback using heuristic entities
    if not triples:
        ents = extract_entities_heuristic(txt)
        if len(ents) >= 2:
            triples.append((ents[0][0], "related_to", ents[1][0]))
    return triples

# -------------------------
# Helpers: build graph, save pyvis
# -------------------------
def build_graph_from_triples_df(df_triples):
    G = nx.DiGraph()
    if df_triples is None or df_triples.empty:
        return G
    # detect columns
    cols = [c.lower() for c in df_triples.columns]
    # try to get subject/object/relation columns
    subj_col = None
    rel_col = None
    obj_col = None
    for c in df_triples.columns:
        cl = c.strip().lower()
        if cl == "subject":
            subj_col = c
        elif cl in ("relation", "verb"):
            rel_col = c
        elif cl == "object":
            obj_col = c
    # fallback names
    if not subj_col or not rel_col or not obj_col:
        # try common names
        mapping = {k.lower(): k for k in df_triples.columns}
        subj_col = mapping.get("subject", df_triples.columns[0] if len(df_triples.columns) >= 1 else None)
        rel_col = mapping.get("relation", df_triples.columns[1] if len(df_triples.columns) >= 2 else None)
        obj_col = mapping.get("object", df_triples.columns[2] if len(df_triples.columns) >= 3 else None)

    for _, r in df_triples.iterrows():
        s = str(r.get(subj_col, "")).strip()
        o = str(r.get(obj_col, "")).strip()
        rel = str(r.get(rel_col, "")).strip()
        if s and o:
            G.add_node(s)
            G.add_node(o)
            G.add_edge(s, o, label=rel)
    return G

def save_pyvis_graph(G, out_html_path):
    net = Network(height="700px", width="100%", directed=True, bgcolor="#ffffff", font_color="#000000")
    net.toggle_physics(True)
    for n in G.nodes():
        deg = max(1, G.degree(n))
        net.add_node(n, label=n, size=12 + deg * 3, color="lightblue")
    for u, v, data in G.edges(data=True):
        lab = data.get("label", "")
        net.add_edge(u, v, title=lab, label=lab)
    out_html_path = str(out_html_path)
    net.save_graph(out_html_path)
    return out_html_path

# -------------------------
# Semantic search helper
# -------------------------
def build_vectorizers(sentences, relation_strings):
    combined = sentences + relation_strings
    if not combined:
        return None, None, None, None
    vectorizer = TfidfVectorizer(stop_words="english").fit(combined)
    sent_vecs = vectorizer.transform(sentences) if sentences else None
    rel_vecs = vectorizer.transform(relation_strings) if relation_strings else None
    return vectorizer, sent_vecs, rel_vecs, combined

# -------------------------
# Page components: each page is a function
# -------------------------
def page_upload():
    st.header("📁 Upload Dataset")
    st.write("Upload a CSV with columns like `domain` and `text`. If you don't upload, a sample will be used.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state["uploaded_df"] = df.copy()
            st.success("Dataset uploaded.")
        except Exception as e:
            st.error("Failed to read CSV: " + str(e))
            st.error(traceback.format_exc())
            return
    else:
        if "uploaded_df" not in st.session_state:
            sample = [
                {"domain":"Finance","text":"Goldman Sachs reported a 10% rise in revenue last quarter."},
                {"domain":"Education","text":"Harvard University launched a new AI program on June 10, 2024."},
                {"domain":"Health","text":"Dr. Sarah Johnson conducted a study on diabetes in 2023."},
                {"domain":"Technology","text":"Google partnered with NASA to advance quantum computing in 2024."}
            ]
            st.session_state["uploaded_df"] = pd.DataFrame(sample)
            st.info("Using sample dataset.")
    df = st.session_state.get("uploaded_df")
    if df is not None:
        st.subheader("Dataset preview (first 20 rows)")
        st.dataframe(df.head(20))
        if st.button("Save dataset to data/processed/cross_domain_dataset.csv"):
            save_path = PROCESSED_DIR / "cross_domain_dataset.csv"
            df.to_csv(save_path, index=False)
            st.success(f"Saved to {save_path}")

def page_preprocessing():
    st.header("🧹 Preprocessing")
    df = st.session_state.get("uploaded_df")
    if df is None:
        st.info("Upload dataset first.")
        return
    st.write("Columns:", list(df.columns))
    st.subheader("Missing values")
    st.dataframe(df.isna().sum())
    if st.button("Strip whitespace from text and lowercase domain names"):
        tcol = None
        for c in df.columns:
            if c.strip().lower() == "text":
                tcol = c
        dcol = None
        for c in df.columns:
            if c.strip().lower() == "domain":
                dcol = c
        if tcol:
            df[tcol] = df[tcol].astype(str).str.strip()
        if dcol:
            df[dcol] = df[dcol].astype(str).str.strip().str.lower()
        st.session_state["uploaded_df"] = df
        st.success("Preprocessing applied.")

def page_ner():
    st.header("🧠 NER (heuristic)")
    df = st.session_state.get("uploaded_df")
    if df is None:
        st.info("Upload dataset first.")
        return
    text_col = None
    for c in df.columns:
        if c.strip().lower() == "text":
            text_col = c
    if not text_col:
        st.error("No 'text' column found.")
        return
    if st.button("Run Heuristic NER"):
        rows = []
        for _, r in df.iterrows():
            txt = str(r[text_col])
            ents = extract_entities_heuristic(txt)
            rows.append({"domain": r.get("domain", ""), "text": txt, "entities": str(ents)})
        ner_df = pd.DataFrame(rows)
        st.session_state["ner_df"] = ner_df
        ner_df.to_csv(ENTITIES_OUT, index=False)
        st.success(f"NER completed — saved to {ENTITIES_OUT}")
    if "ner_df" in st.session_state:
        st.subheader("Entities (sample)")
        st.dataframe(st.session_state["ner_df"].head(50))

def page_relation_extraction():
    st.header("🔎 Relation Extraction (heuristic)")
    df = st.session_state.get("uploaded_df")
    if df is None:
        st.info("Upload dataset first.")
        return
    text_col = None
    for c in df.columns:
        if c.strip().lower() == "text":
            text_col = c
    if not text_col:
        st.error("No 'text' column found.")
        return
    if st.button("Run Relation Extraction"):
        rows = []
        for _, r in df.iterrows():
            txt = str(r[text_col])
            triples = extract_triples_heuristic(txt)
            # convert to dict rows
            for s, rel, o in triples:
                rows.append({"text": txt, "subject": s, "relation": rel, "object": o})
        rel_df = pd.DataFrame(rows)
        st.session_state["rel_df"] = rel_df
        rel_df.to_csv(RELATIONS_OUT, index=False)
        st.success(f"Relations extracted — saved to {RELATIONS_OUT}")
    if "rel_df" in st.session_state:
        st.subheader("Relations (sample)")
        st.dataframe(st.session_state["rel_df"].head(50))

def page_entities_table():
    st.header("📋 Entities Table")
    ner_df = st.session_state.get("ner_df")
    if ner_df is None:
        st.info("Run NER first.")
        return
    st.dataframe(ner_df)

def page_relations_table():
    st.header("📚 Relations Table")
    rel_df = st.session_state.get("rel_df")
    if rel_df is None:
        st.info("Run relation extraction first.")
        return
    st.dataframe(rel_df)

def page_full_graph():
    st.header("🌐 Full Knowledge Graph")
    rel_df = st.session_state.get("rel_df")
    if rel_df is None:
        st.info("Run relation extraction first.")
        return
    st.write("Options:")
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Build & Save Full Graph (PyVis)"):
            G = build_graph_from_triples_df(rel_df)
            path = save_pyvis_graph(G, GRAPH_HTML)
            st.session_state["last_full_graph_html"] = path
            st.success(f"Saved HTML to {path}")
    with c2:
        st.write("After building, open the saved HTML for interactive exploration, or view inline if available below.")
    if "last_full_graph_html" in st.session_state:
        html_path = st.session_state["last_full_graph_html"]
        if os.path.exists(html_path):
            st.components.v1.html(open(html_path, "r", encoding="utf-8").read(), height=700)
        else:
            st.info("Graph file not found — build the graph first.")

def page_semantic_search():
    st.header("🔍 Semantic Search & Subgraph")
    rel_df = st.session_state.get("rel_df")
    df = st.session_state.get("uploaded_df")
    if rel_df is None or df is None:
        st.info("Ensure dataset uploaded and relation extraction run.")
        return
    # prepare corpora
    text_col = None
    for c in df.columns:
        if c.strip().lower() == "text":
            text_col = c
    sentences = df[text_col].astype(str).tolist()
    rs = rel_df
    # build relation strings
    subj_c = find_col(rel_df, ["subject"])
    rel_c = find_col(rel_df, ["relation", "verb"])
    obj_c = find_col(rel_df, ["object"])
    if not subj_c or not rel_c or not obj_c:
        st.error("Relations table missing subject/relation/object columns.")
        return
    relation_strings = rel_df.apply(lambda r: f"{r[subj_c]} {r[rel_c]} {r[obj_c]}", axis=1).tolist()
    vectorizer, sent_vecs, rel_vecs, combined = build_vectorizers(sentences, relation_strings)
    if vectorizer is None:
        st.error("Not enough data to build TF-IDF.")
        return
    query = st.text_input("Enter query (e.g., 'Who launched a program?')")
    n_hits = st.slider("Top N", 1, 10, 3)
    if st.button("Search & Generate Subgraph"):
        if not query.strip():
            st.warning("Enter a query.")
            return
        qv = vectorizer.transform([query])
        sent_scores = linear_kernel(qv, sent_vecs).flatten() if sent_vecs is not None else []
        rel_scores = linear_kernel(qv, rel_vecs).flatten() if rel_vecs is not None else []
        top_sent_idx = sent_scores.argsort()[::-1][:n_hits] if len(sent_scores) else []
        top_rel_idx = rel_scores.argsort()[::-1][:n_hits] if len(rel_scores) else []
        st.subheader("Top matching sentences")
        for i in top_sent_idx:
            st.write(f"- {sentences[i]}  (score {sent_scores[i]:.3f})")
        matched_rels = rel_df.iloc[top_rel_idx]
        st.subheader("Relations used for subgraph")
        st.dataframe(matched_rels)
        # build subgraph
        G_sub = nx.DiGraph()
        for _, r in matched_rels.iterrows():
            s = str(r[subj_c]).strip()
            o = str(r[obj_c]).strip()
            rel = str(r[rel_c]).strip()
            if s and o:
                G_sub.add_node(s)
                G_sub.add_node(o)
                G_sub.add_edge(s, o, label=rel)
        sub_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
        net = Network(height="520px", width="100%", directed=True)
        net.toggle_physics(True)
        for n in G_sub.nodes():
            net.add_node(n, label=n, color="lightblue")
        for u, v, data in G_sub.edges(data=True):
            net.add_edge(u, v, title=data["label"], label=data["label"])
        net.save_graph(sub_html)
        st.components.v1.html(open(sub_html, "r", encoding="utf-8").read(), height=520)
        # store in query_log
        st.session_state.setdefault("query_log", []).append({"query": query, "top_rel_idx": top_rel_idx.tolist()})

def page_subgraph_viewer():
    st.header("🗂 Subgraph Viewer")
    logs = st.session_state.get("query_log", [])
    rel_df = st.session_state.get("rel_df")
    if not logs:
        st.info("No previous semantic searches. Run a search first.")
        return
    choice = st.selectbox("Choose previous query", list(range(len(logs))), format_func=lambda i: f"{i}: {logs[i]['query']}")
    entry = logs[choice]
    idxs = entry.get("top_rel_idx", [])
    matched = rel_df.iloc[idxs]
    st.subheader("Relations for this previous query")
    st.dataframe(matched)
    if st.button("Visualize this subgraph"):
        G_sub = nx.DiGraph()
        s_col = find_col(rel_df, ["subject"])
        r_col = find_col(rel_df, ["relation", "verb"])
        o_col = find_col(rel_df, ["object"])
        for _, r in matched.iterrows():
            s = str(r[s_col]).strip()
            o = str(r[o_col]).strip()
            rel = str(r[r_col]).strip()
            if s and o:
                G_sub.add_node(s)
                G_sub.add_node(o)
                G_sub.add_edge(s, o, label=rel)
        sub_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
        net = Network(height="520px", width="100%", directed=True)
        net.toggle_physics(True)
        for n in G_sub.nodes():
            net.add_node(n, label=n, color="lightblue")
        for u, v, data in G_sub.edges(data=True):
            net.add_edge(u, v, title=data["label"], label=data["label"])
        net.save_graph(sub_html)
        st.components.v1.html(open(sub_html, "r", encoding="utf-8").read(), height=520)

def page_admin():
    st.header("⚙️ Admin Dashboard")
    rel_df = st.session_state.get("rel_df")
    ner_df = st.session_state.get("ner_df")
    df = st.session_state.get("uploaded_df")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Entities", len(ner_df) if ner_df is not None else 0)
    col2.metric("Total Relations", len(rel_df) if rel_df is not None else 0)
    domain_count = 0
    if df is not None:
        for c in df.columns:
            if c.strip().lower() == "domain":
                domain_count = df[c].nunique()
    col3.metric("Unique Domains", domain_count)
    st.subheader("Top Relations")
    if rel_df is not None:
        rcol = find_col(rel_df, ["relation", "verb"])
        if rcol:
            st.bar_chart(rel_df[rcol].value_counts().head(10))
    st.subheader("Logs")
    logs = st.session_state.get("app_logs", [])
    if logs:
        st.dataframe(pd.DataFrame(logs[-30:]))
    else:
        st.info("No logs yet.")
    st.subheader("Graph Cleaning")
    if clean_graph is None:
        st.info("Graph cleaner not found. Add modules/graph_cleaner.py to enable cleaning.")
    else:
        if st.button("Run Graph Cleaning"):
            rel_df = st.session_state.get("rel_df")
            if rel_df is None:
                st.warning("No relations to clean.")
            else:
                try:
                    cleaned = clean_graph(rel_df.copy())
                    st.session_state["rel_df"] = cleaned
                    cleaned.to_csv(PROCESSED_DIR / "relations_cleaned.csv", index=False)
                    st.success("Graph cleaned and saved to data/processed/relations_cleaned.csv")
                except Exception as e:
                    st.error("Cleaning failed: " + str(e))

# -------------------------
# Navigation setup
# -------------------------
PAGES = [
    ("Upload Dataset", page_upload),
    ("Preprocessing", page_preprocessing),
    ("NER Extraction", page_ner),
    ("Relation Extraction", page_relation_extraction),
    ("Entities Table", page_entities_table),
    ("Relations Table", page_relations_table),
    ("Full Knowledge Graph", page_full_graph),
    ("Semantic Search", page_semantic_search),
    ("Subgraph Viewer", page_subgraph_viewer),
    ("Admin Dashboard", page_admin),
]

st.set_page_config(page_title="AI-KnowMap", layout="wide")
st.sidebar.title("AI-KnowMap")
st.sidebar.write("Navigation")
menu = st.sidebar.radio("Go to", [p[0] for p in PAGES])

# Quick actions
if st.sidebar.button("Clear Session"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.experimental_rerun()

# Log page visits
st.session_state.setdefault("app_logs", []).append({"time": pd.Timestamp.now(), "event": f"Visited {menu}"})

# Dispatch
for name, func in PAGES:
    if name == menu:
        func()
        break

st.markdown("---")
st.caption("AI-KnowMap — Cloud-friendly (no spaCy). Developed for Milestone.")
