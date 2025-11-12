# modules/full_app.py
import streamlit as st
import pandas as pd
import os
import ast
import spacy
import networkx as nx
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tempfile

nlp = spacy.load("en_core_web_sm")

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

def build_graph_from_triples(triples_df):
    G = nx.DiGraph()
    for _, r in triples_df.iterrows():
        s = str(r.get("subject","")).strip()
        o = str(r.get("object","")).strip()
        rel = str(r.get("relation","")).strip()
        if not s or not o:
            continue
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
# Semantic search (TF-IDF)
# -------------------
st.header("4) Semantic Search (TF-IDF)")
# create natural language sentences from relations (if available)
sentences = df_texts["text"].astype(str).tolist()  # corpus
vectorizer = TfidfVectorizer(stop_words="english").fit(sentences)
sent_vecs = vectorizer.transform(sentences)

query = st.text_input("Enter a search query (e.g., 'Who launched a program?')", "")
n_hits = st.slider("Results to show", 1, 10, 3)
if st.button("Search"):
    if query.strip() == "":
        st.warning("Type a query first.")
    else:
        qv = vectorizer.transform([query])
        sims = linear_kernel(qv, sent_vecs).flatten()
        top_idx = sims.argsort()[::-1][:n_hits]
        res = [(sentences[i], float(sims[i])) for i in top_idx]
        st.subheader("Top matches")
        for s,score in res:
            st.write(f"- `{s}` — score: {score:.3f}")

# -------------------
# Export & screenshots guidance
# -------------------
st.header("5) Export & Screenshot Guidance")
st.markdown("""
**Screenshots to capture for submission (manually):**
1. **Uploaded dataset view** — The table shown in *Dataset Upload & Preview*.  
2. **Extracted entities/relations table** — The two tables under *Extracted Entities & Relations*.  
3. **Knowledge graph visualization** — Click **Build & Save Interactive Graph** then open `ui/knowledge_graph.html` in browser and screenshot.  
4. **Query-based subgraph** — Run a semantic search query that returns a sentence, then open a subgraph by filtering relations (manually capture the area or implement filter in the code if you want automatic subgraph extraction).
""")

st.markdown("### Saved file locations (inside your project):")
st.write(f"- Entities CSV: `{processed_entities_path}`")
st.write(f"- Relations CSV: `{processed_relations_path}`")
st.write(f"- Interactive graph HTML: `ui/knowledge_graph.html`")

st.success("Prototype integrated. Follow the screenshot guidance above to collect required deliverables.")
