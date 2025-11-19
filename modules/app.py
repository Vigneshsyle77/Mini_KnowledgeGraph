# modules/full_app.py
import streamlit as st
<<<<<<< Updated upstream
import pandas as pd
import os
import ast
import spacy
=======
>>>>>>> Stashed changes
import networkx as nx
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from graph_cleaner import clean_graph
import tempfile

<<<<<<< Updated upstream
import spacy
=======
# -------------------------
# Ensure repo root on path (useful for imports)
# -------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -------------------------
# Optional graph cleaner (if present)
# -------------------------
clean_graph = None
try:
    # prefer modules/graph_cleaner.py
    from modules.graph_cleaner import clean_graph
except Exception:
    try:
        # fallback if running from modules/ directory
        from graph_cleaner import clean_graph
    except Exception:
        clean_graph = None

# -------------------------
# GLOBAL helper functions
# -------------------------
def find_col(df, name_list):
    """
    Auto-detect a column whose lowercase name matches any in name_list.
    Returns actual column name or None.
    """
    if df is None:
        return None
    lower_targets = [x.lower().strip() for x in name_list]
    for col in df.columns:
        if col.lower().strip() in lower_targets:
            return col
    return None

# -------------------------
# Paths
# -------------------------
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
UI_DIR = ROOT / "ui"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
UI_DIR.mkdir(parents=True, exist_ok=True)
>>>>>>> Stashed changes

# Auto-download SpaCy model if not present
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

<<<<<<< Updated upstream
st.set_page_config(page_title="AI-KnowMap — Full Prototype", layout="wide")
st.title("🧠 AI-KnowMap — Full Prototype")

# -------------------
# Sidebar: upload
# -------------------
st.sidebar.header("Upload / Data")
uploaded_texts = st.sidebar.file_uploader("Upload text dataset (CSV)", type=["csv"])
uploaded_entities = st.sidebar.file_uploader("Upload entities.csv (optional)", type=["csv"])
uploaded_relations = st.sidebar.file_uploader("Upload relations.csv (optional)", type=["csv"])
st.sidebar.markdown("---")
run_ner_btn = st.sidebar.button("Run NER & Relation Extraction")

# Internal storage paths (temp)
os.makedirs("data/processed", exist_ok=True)
processed_entities_path = "data/processed/entities_out.csv"
processed_relations_path = "data/processed/relations_out.csv"
pyvis_output_path = "ui/knowledge_graph.html"
os.makedirs("ui", exist_ok=True)

# -------------------
# Helper functions
# -------------------
def run_ner_on_texts(df, text_col="text"):
    rows = []
    for i, txt in enumerate(df[text_col].astype(str)):
        doc = nlp(txt)
        ents = [(ent.text, ent.label_) for ent in doc.ents]
        rows.append({
            "domain": df.columns[0] if "domain" in df.columns else "",
            "text": txt,
            "entities": str(ents)
        })
    out = pd.DataFrame(rows)
    return out
# --- Auto-detect column names (case-insensitive) ---
def find_col(df, name_list):
    for col in df.columns:
        if col.lower().strip() in name_list:
            return col
    return None


def extract_triples_simple(df, text_col="text"):
    # simple subject-verb-object extraction using spaCy dependency parse
    triples = []
    for txt in df[text_col].astype(str):
        doc = nlp(txt)
        # find noun chunks and verbs linking them
        subj = None
        obj = None
        rel = None
        for tok in doc:
            if tok.dep_.endswith("subj") and tok.head.pos_ == "VERB":
                subj = tok.text
                rel = tok.head.lemma_
                # try to find object of verb
                for child in tok.head.children:
                    if child.dep_.endswith("obj") or child.dep_.lower() in ("dobj","pobj"):
                        obj = child.text
                        break
                if subj and rel and obj:
                    triples.append({"text": txt, "subject": subj, "relation": rel, "object": obj})
        # fallback: if no trip found, try NER-based pairing (entity + nearest DATE/ORG)
        if not triples or len(triples) < 1:
            ents = [(ent.text, ent.label_) for ent in doc.ents]
            if len(ents) >= 2:
                # pair first two
                triples.append({"text": txt, "subject": ents[0][0], "relation": "related_to", "object": ents[1][0]})
    return pd.DataFrame(triples)

def build_graph_from_triples(df):
    G = nx.DiGraph()

    # Auto-detect column names
    subject_col = find_col(df, ["subject"])
    relation_col = find_col(df, ["relation"])
    object_col  = find_col(df, ["object"])

    if not subject_col or not relation_col or not object_col:
        st.error("❌ Could not find subject / relation / object in relations file.")
        st.warning(f"Detected columns: {list(df.columns)}")
        return G

    for _, r in df.iterrows():
        s = str(r[subject_col]).strip()
        o = str(r[object_col]).strip()
        rel = str(r[relation_col]).strip()

=======
# -------------------------
# Heuristic NER / patterns (no spaCy)
# -------------------------
ORG_KEYWORDS = [
    "University", "Inc", "Corp", "Corporation", "Company", "Google", "Amazon", "NASA",
    "Microsoft", "Apple", "Services", "Institute", "Bank", "Harvard", "Stanford",
    "John Hopkins", "Goldman Sachs", "AWS", "IBM", "Meta"
]
PERSON_TITLE_PATTERN = r'\b(Dr|Mr|Ms|Mrs|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'
YEAR_PATTERN = r'\b(19|20)\d{2}\b'
MONTH_PATTERN = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b'
PERCENT_PATTERN = r'\b\d+%'

def extract_entities_heuristic(text):
    """
    Returns a list of (entity_text, label) using simple heuristics.
    """
    ents = []
    if not isinstance(text, str) or text.strip() == "":
        return ents

    # DATE: years
    for m in re.finditer(YEAR_PATTERN, text):
        ents.append((m.group(0), "DATE"))

    # DATE: months (+ optional day/year)
    for m in re.finditer(MONTH_PATTERN + r'(?:\s+\d{1,2},\s*\d{4})?', text):
        ents.append((m.group(0), "DATE"))

    # PERCENT
    for m in re.finditer(PERCENT_PATTERN, text):
        ents.append((m.group(0), "PERCENT"))

    # PERSON via title
    for m in re.finditer(PERSON_TITLE_PATTERN, text):
        ents.append((m.group(0), "PERSON"))

    # PERSON via two consecutive Titlecase words
    two_caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
    for t in two_caps:
        if any(k.lower() in t.lower() for k in ORG_KEYWORDS):
            continue
        if len(t.split()) <= 3:
            ents.append((t, "PERSON"))

    # ORG via keywords
    for k in ORG_KEYWORDS:
        for m in re.finditer(r'\b' + re.escape(k) + r'\b', text):
            ents.append((m.group(0), "ORG"))

    # ORG via all-caps acronyms
    for m in re.finditer(r'\b([A-Z]{2,10})\b', text):
        token = m.group(1)
        if token.isupper() and len(token) >= 2:
            ents.append((token, "ORG"))

    # Deduplicate preserving order
    seen = set()
    filtered = []
    for e, label in ents:
        key = (e.strip(), label)
        if key not in seen and e.strip():
            seen.add(key)
            filtered.append((e.strip(), label))
    return filtered

# -------------------------
# Heuristic relation extraction
# -------------------------
REL_KEYWORDS = [
    "launched", "reported", "conducted", "collaborated", "partnered",
    "released", "held", "invested", "announced", "founded", "created", "reported"
]

def extract_triples_heuristic(text):
    """
    Return list of triples (subject, relation, object) detected heuristically.
    """
    triples = []
    if not isinstance(text, str):
        return triples
    txt = text.strip()
    low = txt.lower()
    for rel_kw in REL_KEYWORDS:
        if rel_kw in low:
            idx = low.find(rel_kw)
            before = txt[:idx].strip()
            after = txt[idx + len(rel_kw):].strip()
            # Subject heuristic: last TitleCase chunk in 'before'
            match_subj = re.findall(r'([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)*)', before)
            if match_subj:
                subj = match_subj[-1]
            else:
                subj = before.split()[-1] if before.split() else before
            # Object heuristic: first TitleCase chunk in 'after'
            match_obj = re.findall(r'([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)*)', after)
            if match_obj:
                obj = match_obj[0]
            else:
                obj = after.split()[0] if after.split() else after
            triples.append((subj.strip(), rel_kw, obj.strip()))
    # Fallback: pair first two entities if no rel found
    if not triples:
        ents = extract_entities_heuristic(txt)
        if len(ents) >= 2:
            triples.append((ents[0][0], "related_to", ents[1][0]))
    return triples

# -------------------------
# Graph helpers
# -------------------------
def build_graph_from_triples_df(df_triples):
    G = nx.DiGraph()
    if df_triples is None or df_triples.empty:
        return G
    # auto-detect columns
    subj_col = find_col(df_triples, ["subject"])
    rel_col = find_col(df_triples, ["relation", "verb"])
    obj_col = find_col(df_triples, ["object"])
    # fallback column mapping
    if not subj_col or not rel_col or not obj_col:
        mapping = {k.lower(): k for k in df_triples.columns}
        subj_col = subj_col or mapping.get("subject", df_triples.columns[0] if len(df_triples.columns) > 0 else None)
        rel_col = rel_col or mapping.get("relation", df_triples.columns[1] if len(df_triples.columns) > 1 else None)
        obj_col = obj_col or mapping.get("object", df_triples.columns[2] if len(df_triples.columns) > 2 else None)
    for _, r in df_triples.iterrows():
        try:
            s = str(r.get(subj_col, "")).strip()
            o = str(r.get(obj_col, "")).strip()
            rel = str(r.get(rel_col, "")).strip()
        except Exception:
            continue
>>>>>>> Stashed changes
        if s and o:
            G.add_node(s)
            G.add_node(o)
            G.add_edge(s, o, label=rel)

    return G



def save_pyvis(G, output_html=pyvis_output_path):
    net = Network(height="750px", width="100%", directed=True, bgcolor="#ffffff", font_color="#000000")
    net.toggle_physics(True)
    for n, data in G.nodes(data=True):
        net.add_node(n, label=n, color="lightblue")
    for u,v,data in G.edges(data=True):
        label = data.get("label","")
        net.add_edge(u,v,title=label,label=label)
    net.save_graph(output_html)
    return output_html

<<<<<<< Updated upstream
# -------------------
# Load uploaded / fallback
# -------------------
st.header("1) Dataset Upload & Preview")

if uploaded_texts:
    df_texts = pd.read_csv(uploaded_texts)
    st.success("Uploaded dataset loaded.")
else:
    # fallback example small built-in dataset
    sample = [
        {"domain":"Finance","text":"Goldman Sachs reported a 10% rise in revenue last quarter."},
        {"domain":"Education","text":"Harvard University launched a new AI program on June 10, 2024."},
        {"domain":"Health","text":"Dr. Sarah Johnson conducted a study on diabetes in 2023."},
        {"domain":"Technology","text":"Google partnered with NASA to advance quantum computing in 2024."},
    ]
    df_texts = pd.DataFrame(sample)
    st.info("No file uploaded — using sample dataset for demo. Upload a CSV with columns (domain,text) to use your data.")

st.dataframe(df_texts.head(20))

# allow uploading pre-extracted entities/relations
if uploaded_entities:
    df_entities = pd.read_csv(uploaded_entities)
    st.sidebar.success("Loaded entities file")
else:
    df_entities = None

if uploaded_relations:
    df_relations = pd.read_csv(uploaded_relations)
    st.sidebar.success("Loaded relations file")
else:
    df_relations = None

# Run NER and relation extraction if requested (or if no uploaded files)
if run_ner_btn or (df_entities is None or df_relations is None):
    with st.spinner("Running NER and relation extraction (spaCy)..."):
        ner_out = run_ner_on_texts(df_texts, text_col="text")
        ner_out.to_csv(processed_entities_path, index=False)
        st.success(f"Entities extracted — saved to {processed_entities_path}")
        rels_out = extract_triples_simple(df_texts, text_col="text")
        rels_out.to_csv(processed_relations_path, index=False)
        st.success(f"Relations extracted — saved to {processed_relations_path}")
    df_entities = ner_out
    df_relations = rels_out

# Show extracted entities/relations
st.header("2) Extracted Entities & Relations")
colA, colB = st.columns(2)
with colA:
    st.subheader("Entities (sample)")
    st.dataframe(df_entities.head(20))
with colB:
    st.subheader("Relations (sample)")
    st.dataframe(df_relations.head(20))

# Save copies for UI
df_entities.to_csv(processed_entities_path, index=False)
df_relations.to_csv(processed_relations_path, index=False)

st.header("🧹  Graph Cleaning Module")

if st.button("Run Graph Cleaning"):
    df_relations_cleaned = clean_graph(df_relations)

    df_relations_cleaned.to_csv("data/processed/relations_cleaned.csv", index=False)
    st.success("Graph cleaned! Saved as relations_cleaned.csv")

    st.subheader("Cleaned Relations Preview")
    st.dataframe(df_relations_cleaned.head(20))

# -------------------
# Build graph and visualise
# -------------------
st.header("3) Knowledge Graph Visualization")
if st.button("Build & Save Interactive Graph (PyVis)"):
    G = build_graph_from_triples(df_relations)
    out_html = save_pyvis(G)
    st.success(f"Saved interactive graph to {out_html}")
    st.markdown(f"Open the generated graph file: `{out_html}` in your browser (or use the UI button below).")

# Show quick static summary & option to open HTML
G_preview = build_graph_from_triples(df_relations)
st.write(f"Graph: {len(G_preview.nodes())} nodes — {len(G_preview.edges())} edges")
if os.path.exists(pyvis_output_path):
    st.markdown(f"[Open saved interactive graph]({pyvis_output_path})", unsafe_allow_html=True)
else:
    st.info("Build graph and Save first to generate interactive HTML.")

# -------------------
# 4) Semantic Search + Guaranteed Working Subgraph
# -------------------
sentences = df_texts["text"].astype(str).tolist()
st.header("4) Semantic Search & Subgraph Visualization")

# --- Auto-detect relation columns (case insensitive) ---
def find_col(df, name_list):
    for col in df.columns:
        if col.lower().strip() in name_list:
            return col
    return None

subject_col = find_col(df_relations, ["subject"])
relation_col = find_col(df_relations, ["relation", "verb"])
object_col  = find_col(df_relations, ["object"])

# If missing ANY column, warn & stop semantic subgraph extraction
if not subject_col or not relation_col or not object_col:
    st.error("❌ Relations file does not have subject / relation / object columns.")
    st.warning(f"Detected columns: {list(df_relations.columns)}")
    st.stop()

# Prepare TF-IDF on BOTH sentences and relation metadata
relation_strings = df_relations.apply(
    lambda r: f"{r[subject_col]} {r[relation_col]} {r[object_col]}", axis=1
).tolist()


combined_corpus = sentences + relation_strings
vectorizer = TfidfVectorizer(stop_words="english").fit(combined_corpus)

sent_vecs = vectorizer.transform(sentences)
rel_vecs = vectorizer.transform(relation_strings)

query = st.text_input("Enter a search query (e.g., 'Who launched a program?')")
n_hits = st.slider("Results to show", 1, 10, 3)

if st.button("🔍 Search & Build Subgraph"):
    if not query.strip():
        st.warning("Enter a query first.")
    else:
=======
# -------------------------
# Semantic helpers
# -------------------------
def build_vectorizers(sentences, relation_strings):
    combined = (sentences or []) + (relation_strings or [])
    if not combined:
        return None, None, None, None
    vectorizer = TfidfVectorizer(stop_words="english").fit(combined)
    sent_vecs = vectorizer.transform(sentences) if sentences else None
    rel_vecs = vectorizer.transform(relation_strings) if relation_strings else None
    return vectorizer, sent_vecs, rel_vecs, combined

# -------------------------
# Page functions
# -------------------------
def page_upload():
    st.header("📁 Upload Dataset")
    st.write("Upload a CSV with columns like `domain` and `text`. If you don't upload, a sample will be used.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state["uploaded_df"] = df.copy()
            st.success("Dataset uploaded and stored in session.")
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
            st.info("Using built-in sample dataset. Upload your CSV to replace it.")
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
    st.write("Columns detected:", list(df.columns))
    st.subheader("Missing values")
    st.dataframe(df.isna().sum())
    if st.button("Strip whitespace from text and lowercase domain names"):
        tcol = find_col(df, ["text"])
        dcol = find_col(df, ["domain"])
        if tcol:
            df[tcol] = df[tcol].astype(str).str.strip()
        if dcol:
            df[dcol] = df[dcol].astype(str).str.strip().str.lower()
        st.session_state["uploaded_df"] = df
        st.success("Preprocessing applied and saved to session.")

def page_ner():
    st.header("🧠 NER (heuristic)")
    df = st.session_state.get("uploaded_df")
    if df is None:
        st.info("Upload dataset first.")
        return
    text_col = find_col(df, ["text"])
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
    text_col = find_col(df, ["text"])
    if not text_col:
        st.error("No 'text' column found.")
        return
    if st.button("Run Relation Extraction"):
        rows = []
        for _, r in df.iterrows():
            txt = str(r[text_col])
            triples = extract_triples_heuristic(txt)
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
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Build & Save Full Graph (PyVis)"):
            G = build_graph_from_triples_df(rel_df)
            path = save_pyvis_graph(G, GRAPH_HTML)
            st.session_state["last_full_graph_html"] = path
            st.success(f"Saved HTML to {path}")
    with c2:
        st.write("After building, open the saved HTML for interactive exploration, or view inline below if available.")
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
    text_col = find_col(df, ["text"])
    if not text_col:
        st.error("No text column in dataset.")
        return
    sentences = df[text_col].astype(str).tolist()
    subj_c = find_col(rel_df, ["subject"])
    rel_c = find_col(rel_df, ["relation", "verb"])
    obj_c = find_col(rel_df, ["object"])
    if not subj_c or not rel_c or not obj_c:
        st.error("Relations table missing subject/relation/object columns.")
        return
    relation_strings = rel_df.apply(lambda r: f"{r[subj_c]} {r[rel_c]} {r[obj_c]}", axis=1).tolist()
    vectorizer, sent_vecs, rel_vecs, combined = build_vectorizers(sentences, relation_strings)
    if vectorizer is None:
        st.error("Not enough data for TF-IDF vectorizer.")
        return
    query = st.text_input("Enter query (e.g., 'Who launched a program?')")
    n_hits = st.slider("Top N results", 1, 10, 3)
    if st.button("Search & Generate Subgraph"):
        if not query.strip():
            st.warning("Enter a query.")
            return
>>>>>>> Stashed changes
        qv = vectorizer.transform([query])

        # Rank SENTENCES by similarity
        sent_scores = linear_kernel(qv, sent_vecs).flatten()
        top_sent_idx = sent_scores.argsort()[::-1][:n_hits]

        # Rank RELATIONS by similarity
        rel_scores = linear_kernel(qv, rel_vecs).flatten()
        top_rel_idx = rel_scores.argsort()[::-1][:n_hits]

        # ----------------------------------
        # SHOW TOP SENTENCE MATCHES
        # ----------------------------------
        st.subheader("📌 Top Matching Sentences")
        for idx in top_sent_idx:
            st.write(f"**{sentences[idx]}**  — (score {sent_scores[idx]:.3f})")

        # ----------------------------------
        # COLLECT SUBGRAPH TRIPLES
        # ----------------------------------
        sub_rels = df_relations.iloc[top_rel_idx]

        st.subheader("🔗 Relations Used for Subgraph")
        st.dataframe(sub_rels)

        # ----------------------------------
        # BUILD & VISUALIZE SUBGRAPH
        # ----------------------------------
        G_sub = nx.DiGraph()

        for _, r in sub_rels.iterrows():
            s = str(r[subject_col]).strip()
            o = str(r[object_col]).strip()
            rel = str(r[relation_col]).strip()

            G_sub.add_node(s)
            G_sub.add_node(o)
            G_sub.add_edge(s, o, label=rel)

        # Render PyVis
        subgraph_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
        net = Network(height="600px", width="100%", directed=True,
                      bgcolor="#ffffff", font_color="#000000")

        for node in G_sub.nodes():
            net.add_node(node, label=node, color="lightblue")

        for u, v, data in G_sub.edges(data=True):
<<<<<<< Updated upstream
            net.add_edge(u, v, label=data["label"], title=data["label"])

        net.save_graph(subgraph_html)
        st.components.v1.html(open(subgraph_html, "r", encoding="utf-8").read(), height=600)

# -------------------
# Export & Screenshot Guidance
# -------------------
st.header("5) Export & Screenshot Guidance")

st.markdown("""
Below are the screenshots you must include in the submission.
=======
            net.add_edge(u, v, title=data.get("label", ""), label=data.get("label", ""))
        net.save_graph(sub_html)
        st.components.v1.html(open(sub_html, "r", encoding="utf-8").read(), height=520)
        # log query
        st.session_state.setdefault("query_log", []).append({"query": query, "top_rel_idx": top_rel_idx.tolist()})

def page_subgraph_viewer():
    st.header("🗂 Subgraph Viewer (previous queries)")
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
            net.add_edge(u, v, title=data.get("label", ""), label=data.get("label", ""))
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
        dcol = find_col(df, ["domain"])
        if dcol:
            domain_count = df[dcol].nunique()
    col3.metric("Unique Domains", domain_count)
    st.subheader("Top Relations")
    if rel_df is not None:
        rcol = find_col(rel_df, ["relation", "verb"])
        if rcol:
            st.bar_chart(rel_df[rcol].value_counts().head(10))
    st.subheader("Recent Logs")
    logs = st.session_state.get("app_logs", [])
    if logs:
        st.dataframe(pd.DataFrame(logs[-40:]))
    else:
        st.info("No app logs yet.")
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
                    st.error(traceback.format_exc())

# -------------------------
# Navigation & layout
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
>>>>>>> Stashed changes

### **1. Uploaded Dataset View**
""")
st.image("images/dataset_preview.png",
         caption="Dataset Preview Screenshot",
         use_container_width=True)

st.markdown("""
### **2. Extracted Entities & Relations Table**
""")
st.image("images/entities_relations_table.png",
         caption="Entities & Relations Table Screenshot",
         use_container_width=True)

<<<<<<< Updated upstream
st.markdown("""
### **3. Full Knowledge Graph Visualization**
""")
st.image("images/full_graph.png",
         caption="Full Knowledge Graph Screenshot",
         use_container_width=True)

st.markdown("""
### **4. Semantic Search Results**
""")
st.image("images/semantic_search_results.png",
         caption="Semantic Search Results Screenshot",
         use_container_width=True)
=======
# App logs
st.session_state.setdefault("app_logs", []).append({"time": pd.Timestamp.now(), "event": f"Visited {menu}"})

# Dispatch selected page
for name, func in PAGES:
    if name == menu:
        func()
        break
>>>>>>> Stashed changes

st.markdown("""
### **5. Subgraph Generated from Query**
""")
st.image("images/subgraph.png",
         caption="Semantic Subgraph Screenshot",
         use_container_width=True)


# st.success("All expected screenshots have been listed. Please include them in your final submission.")

st.markdown("### Saved file locations (inside your project):")
st.write(f"- Entities CSV: `{processed_entities_path}`")
st.write(f"- Relations CSV: `{processed_relations_path}`")
st.write(f"- Interactive graph HTML: `ui/knowledge_graph.html`")

st.success("Prototype integrated. Follow the screenshot guidance above to collect required deliverables.")
