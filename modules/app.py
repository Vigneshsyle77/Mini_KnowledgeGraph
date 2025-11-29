# modules/app.py
"""
AI-KnowMap — Neon Themed Knowledge Graph Studio
No spaCy. Heuristic NER + Relation Extraction + Semantic Subgraph.

Run from repo root:
    streamlit run modules/app.py
"""

import os
import sys
import re
import time
import tempfile
import hashlib
from pathlib import Path
import traceback

import pandas as pd
import streamlit as st
import networkx as nx
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ------------------------------------------------------------------------------
# PATHS / ROOT
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
UI_DIR = ROOT / "ui"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
UI_DIR.mkdir(parents=True, exist_ok=True)

ENTITIES_OUT = PROCESSED_DIR / "entities_out.csv"
RELATIONS_OUT = PROCESSED_DIR / "relations_out.csv"
GRAPH_HTML = UI_DIR / "knowledge_graph.html"
USERS_CSV = PROCESSED_DIR / "users.csv"

# optional graph cleaner (if you later add modules/graph_cleaner.py)
try:
    from modules.graph_cleaner import clean_graph
except Exception:
    try:
        from graph_cleaner import clean_graph
    except Exception:
        clean_graph = None

# ------------------------------------------------------------------------------
# STREAMLIT PAGE CONFIG + THEME
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-KnowMap Studio",
    layout="wide",
    page_icon="🧠",
)

def inject_global_theme():
    """Neon gradient + particle background + unified button / card styles."""
    st.markdown(
        """
        <style>
        /* App background: gradient + particles */
        [data-testid="stAppViewContainer"] {
            background:
              radial-gradient(circle at 10% 20%, rgba(0, 255, 240, 0.18) 0, transparent 40%),
              radial-gradient(circle at 80% 10%, rgba(255, 0, 212, 0.16) 0, transparent 45%),
              radial-gradient(circle at 0% 80%, rgba(0, 180, 255, 0.16) 0, transparent 40%),
              linear-gradient(135deg, #05060a 0%, #02050d 40%, #030c18 100%);
            background-size: 200% 200%;
            animation: moveGradient 25s ease alternate infinite;
            color: #f5f5f5;
        }

        /* subtle particle-like dots */
        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
              radial-gradient(circle, rgba(255,255,255,0.08) 1px, transparent 1px),
              radial-gradient(circle, rgba(0,255,255,0.08) 1px, transparent 1px);
            background-size: 120px 120px, 200px 200px;
            opacity: 0.4;
            mix-blend-mode: screen;
            animation: driftDots 40s linear infinite;
            z-index: -1;
        }

        @keyframes moveGradient {
            0% { background-position: 0% 0%; }
            50% { background-position: 50% 100%; }
            100% { background-position: 100% 0%; }
        }

        @keyframes driftDots {
            0% { transform: translate3d(0, 0, 0); }
            100% { transform: translate3d(-80px, -60px, 0); }
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: rgba(5, 10, 25, 0.98);
            backdrop-filter: blur(18px);
            border-right: 1px solid rgba(0, 255, 255, 0.35);
            box-shadow: 8px 0 24px rgba(0, 0, 0, 0.6);
        }
        [data-testid="stSidebar"] * {
            color: #e5e7f1 !important;
        }
        .sidebar-logo {
            font-weight: 800;
            font-size: 22px;
            letter-spacing: 0.06em;
            margin-bottom: 0.25rem;
            background: linear-gradient(90deg, #00eaff, #ff00d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .sidebar-sub {
            font-size: 11px;
            opacity: 0.7;
            margin-bottom: 0.75rem;
        }

        /* Buttons */
        .stButton>button {
            width: 100%;
            border-radius: 999px;
            border: 1px solid rgba(0, 234, 255, 0.6);
            background: radial-gradient(circle at 20% 0%, #ff00d420 0, transparent 60%),
                        linear-gradient(90deg, #02111d, #041b29);
            color: #f8fafd;
            padding: 0.45rem 1.2rem;
            font-weight: 600;
            font-size: 0.9rem;
            box-shadow: 0 0 0 rgba(0, 234, 255, 0.0);
            transition: all 0.22s ease-out;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 0 18px rgba(0, 234, 255, 0.45);
            border-color: rgba(255, 0, 212, 0.65);
        }

        /* Metrics / cards */
        .glass-card {
            padding: 1rem 1.4rem 1.1rem 1.4rem;
            border-radius: 18px;
            background: radial-gradient(circle at top left, rgba(0, 234, 255, 0.08), transparent 55%),
                        rgba(6, 12, 32, 0.96);
            border: 1px solid rgba(255, 255, 255, 0.03);
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.65);
        }
        .neon-title {
            font-size: 30px;
            font-weight: 800;
            margin-bottom: 0.2rem;
            text-shadow: 0 0 14px rgba(0, 234, 255, 0.8);
        }
        .section-label {
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #9ea6ff;
            opacity: 0.85;
        }

        h1, h2, h3 {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_global_theme()

# ------------------------------------------------------------------------------
# SIMPLE LOCAL AUTH (users.csv)
# ------------------------------------------------------------------------------

def hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()


def create_user(email: str, pwd: str) -> bool:
    """Create a user in users.csv with columns: email, password_hash."""
    h = hash_password(pwd)
    if USERS_CSV.exists():
        df = pd.read_csv(USERS_CSV)
        if "email" in df.columns and email in df["email"].values:
            return False
        df.loc[len(df)] = [email, h]
        df.to_csv(USERS_CSV, index=False)
    else:
        pd.DataFrame([{"email": email, "password_hash": h}]).to_csv(
            USERS_CSV, index=False
        )
    return True


def verify_user(email: str, pwd: str) -> bool:
    """Return True if email/password matches row in users.csv."""
    if not USERS_CSV.exists():
        return False
    df = pd.read_csv(USERS_CSV)
    if "email" not in df.columns or "password_hash" not in df.columns:
        return False
    h = hash_password(pwd)
    row = df[(df["email"] == email) & (df["password_hash"] == h)]
    return len(row) > 0

# ------------------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------------------

def find_col(df: pd.DataFrame, name_list):
    """Auto-detect a column whose name (lowercased) is in name_list."""
    if df is None or df.empty:
        return None
    targets = [n.lower().strip() for n in name_list]
    for col in df.columns:
        if col.lower().strip() in targets:
            return col
    return None


def themed_header(title: str, icon: str = "🧠"):
    st.markdown(
        f"""
        <div class="glass-card" style="margin-bottom:1.1rem;">
            <div class="section-label">{icon} MODULE</div>
            <div class="neon-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_loader(message: str = "Processing..."):
    """Neon animated loader used around heavy operations."""
    with st.spinner(message):
        st.markdown(
            """
            <div style="padding:0.5rem 0.1rem 0.8rem 0.1rem;">
              <div style="
                   height:4px;
                   width:100%;
                   border-radius:999px;
                   background:linear-gradient(90deg,#00eaff,#ff00d4,#00eaff);
                   background-size:200% 100%;
                   animation: loaderBar 1.2s linear infinite;">
              </div>
            </div>
            <style>
            @keyframes loaderBar {
              0% { background-position:0% 0%; }
              100% { background-position:200% 0%; }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.6)

# ------------------------------------------------------------------------------
# HEURISTIC NER (NO SPACY)
# ------------------------------------------------------------------------------

ORG_KEYWORDS = [
    "University", "Inc", "Corp", "Corporation", "Company", "Google", "Amazon",
    "NASA", "Microsoft", "Apple", "Services", "Institute", "Bank",
    "Harvard", "Stanford", "John Hopkins", "Goldman Sachs",
]

PERSON_TITLE_PATTERN = r'\b(Dr|Mr|Ms|Mrs|Prof)\.?\s+[A-Z][a-z]+\s?[A-Z]?[a-z]*\b'
YEAR_PATTERN = r'\b(19|20)\d{2}\b'
MONTH_PATTERN = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b'
PERCENT_PATTERN = r'\b\d+%'


def extract_entities_heuristic(text: str):
    ents = []
    if not isinstance(text, str) or not text.strip():
        return ents

    # DATE: years
    for m in re.finditer(YEAR_PATTERN, text):
        ents.append((m.group(0), "DATE"))

    # DATE: month + optional day/year
    for m in re.finditer(MONTH_PATTERN + r'(?:\s+\d{1,2},\s*\d{4})?', text):
        ents.append((m.group(0), "DATE"))

    # PERCENT
    for m in re.finditer(PERCENT_PATTERN, text):
        ents.append((m.group(0), "PERCENT"))

    # PERSON via title
    for m in re.finditer(PERSON_TITLE_PATTERN, text):
        ents.append((m.group(0), "PERSON"))

    # PERSON via two consecutive capitalised words (avoid org keywords)
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

    # All-caps acronyms
    for m in re.finditer(r'\b([A-Z]{2,10})\b', text):
        token = m.group(1)
        if 2 <= len(token) <= 10:
            ents.append((token, "ORG"))

    # De-dupe
    seen = set()
    out = []
    for e, label in ents:
        key = (e.strip(), label)
        if key not in seen and e.strip():
            seen.add(key)
            out.append((e.strip(), label))
    return out

# ------------------------------------------------------------------------------
# HEURISTIC RELATION EXTRACTION
# ------------------------------------------------------------------------------

REL_KEYWORDS = [
    "launched", "reported", "conducted", "collaborated", "partnered",
    "released", "held", "invested", "announced", "founded", "created",
    "introduced",
]


def extract_triples_heuristic(text: str):
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

            # subject: last capitalised phrase before verb
            match_subj = re.findall(r'([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)*)', before)
            if match_subj:
                subj = match_subj[-1]
            else:
                subj = before.split()[-1] if before.split() else before

            # object: first capitalised phrase after verb
            match_obj = re.findall(r'([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)*)', after)
            if match_obj:
                obj = match_obj[0]
            else:
                obj = after.split()[0] if after.split() else after

            triples.append((subj.strip(), rel_kw, obj.strip()))

    # fallback: first two entities
    if not triples:
        ents = extract_entities_heuristic(txt)
        if len(ents) >= 2:
            triples.append((ents[0][0], "related_to", ents[1][0]))

    return triples

# ------------------------------------------------------------------------------
# GRAPH & SEMANTIC HELPERS
# ------------------------------------------------------------------------------

def build_graph_from_triples_df(df_triples: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    if df_triples is None or df_triples.empty:
        return G

    subj_col = find_col(df_triples, ["subject"])
    rel_col = find_col(df_triples, ["relation", "verb"])
    obj_col = find_col(df_triples, ["object"])

    if not subj_col or not rel_col or not obj_col:
        return G

    for _, r in df_triples.iterrows():
        s = str(r[subj_col]).strip()
        o = str(r[obj_col]).strip()
        rel = str(r[rel_col]).strip()
        if s and o:
            G.add_node(s)
            G.add_node(o)
            G.add_edge(s, o, label=rel)
    return G


def save_pyvis_graph(G: nx.DiGraph, out_path: Path) -> str:
    net = Network(
        height="650px",
        width="100%",
        directed=True,
        bgcolor="#020308",
        font_color="#f6f7ff",
    )
    net.toggle_physics(True)

    for n in G.nodes():
        deg = max(1, G.degree(n))
        net.add_node(
            n,
            label=n,
            size=12 + deg * 3,
            color="#00eaff",
            borderWidth=2,
        )

    for u, v, data in G.edges(data=True):
        label = data.get("label", "")
        net.add_edge(
            u,
            v,
            title=label,
            label=label,
            color="#ff00d4",
            width=1.4,
        )

    net.save_graph(str(out_path))
    return str(out_path)


def build_vectorizers(sentences, relation_strings):
    combined = sentences + relation_strings
    if not combined:
        return None, None, None, None
    vec = TfidfVectorizer(stop_words="english").fit(combined)
    sent_vecs = vec.transform(sentences) if sentences else None
    rel_vecs = vec.transform(relation_strings) if relation_strings else None
    return vec, sent_vecs, rel_vecs, combined

# ------------------------------------------------------------------------------
# AUTH PAGE
# ------------------------------------------------------------------------------

def page_auth():
    st.header("🔐 User Authentication")

    tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        pwd = st.text_input("Password", type="password", key="login_pwd")
        remember = st.checkbox("Remember me", key="login_remember")

        if st.button("Login"):
            if verify_user(email, pwd):
                st.session_state["logged_in"] = True
                st.session_state["user_email"] = email
                if remember:
                    st.session_state["remember"] = True
                st.success("Login successful 🎉")
                st.rerun()
            else:
                st.error("Invalid email or password")

    with tab_signup:
        new_email = st.text_input("New Email", key="signup_email")
        new_pwd = st.text_input("Create Password", type="password", key="signup_pwd")

        if st.button("Create Account"):
            if not new_email or not new_pwd:
                st.error("Please enter both email and password.")
            else:
                if create_user(new_email, new_pwd):
                    st.success("Account created. Please login.")
                else:
                    st.error("This user already exists.")

# ------------------------------------------------------------------------------
# WELCOME PAGE
# ------------------------------------------------------------------------------

def page_welcome():
    themed_header("Welcome to AI-KnowMap Studio", "✨")
    st.markdown(
        """
        <div style="padding:1rem 0;">
          <p style="font-size:1.02rem;max-width:780px;">
            Explore, clean, and search a cross-domain knowledge graph built from real-world text.
            Upload datasets, extract entities, visualize graphs, and generate semantic subgraphs —
            all inside a single neon studio.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
            <div class="glass-card">
              <div class="section-label">Onboarding</div>
              <h3 style="margin-top:0.4rem;margin-bottom:0.6rem;">How to get started</h3>
              <ol style="line-height:1.7;font-size:0.95rem;">
                <li>Go to <b>Upload Dataset</b> and load your CSV (or use the sample dataset).</li>
                <li>Run <b>NER Extraction</b> and <b>Relation Extraction</b>.</li>
                <li>Open <b>Full Knowledge Graph</b> to view the global graph.</li>
                <li>Use <b>Semantic Search</b> to generate focused subgraphs.</li>
                <li>Monitor stats from the <b>Admin Dashboard</b>.</li>
              </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="glass-card">
              <div class="section-label">Status</div>
              <h3 style="margin-top:0.4rem;">Pipeline Snapshot</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        df = st.session_state.get("uploaded_df")
        ner_df = st.session_state.get("ner_df")
        rel_df = st.session_state.get("rel_df")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Sentences", len(df) if df is not None else 0)
        with c2:
            st.metric("Relations", len(rel_df) if rel_df is not None else 0)
        st.metric("Entities", len(ner_df) if ner_df is not None else 0)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Use the navigation menu on the left to move between modules.")

# ------------------------------------------------------------------------------
# PIPELINE PAGES
# ------------------------------------------------------------------------------

def page_upload():
    themed_header("Upload & Inspect Dataset", "📁")

    st.write("Upload a CSV with at least a `text` column (and optionally `domain`).")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state["uploaded_df"] = df.copy()
            st.success("Dataset uploaded successfully.")
        except Exception as e:
            st.error("Failed to read CSV: " + str(e))
            st.error(traceback.format_exc())
            return
    else:
        if "uploaded_df" not in st.session_state:
            sample = [
                {"domain": "Finance",
                 "text": "Goldman Sachs reported a 10% rise in revenue last quarter."},
                {"domain": "Education",
                 "text": "Harvard University launched a new AI program on June 10, 2024."},
                {"domain": "Health",
                 "text": "Dr. Sarah Johnson conducted a study on diabetes in 2023."},
                {"domain": "Technology",
                 "text": "Google partnered with NASA to advance quantum computing in 2024."},
            ]
            st.session_state["uploaded_df"] = pd.DataFrame(sample)
            st.info("No file uploaded — using a small built-in sample dataset.")

    df = st.session_state.get("uploaded_df")
    if df is not None:
        st.subheader("Dataset Preview (first 20 rows)")
        st.dataframe(df.head(20), use_container_width=True)

        if st.button("💾 Save dataset → data/processed/cross_domain_dataset.csv"):
            save_path = PROCESSED_DIR / "cross_domain_dataset.csv"
            df.to_csv(save_path, index=False)
            st.success(f"Saved to {save_path}")


def page_preprocessing():
    themed_header("Preprocessing & Data Health", "🧹")
    df = st.session_state.get("uploaded_df")
    if df is None:
        st.info("Upload a dataset first.")
        return

    st.write("Columns detected:", list(df.columns))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Missing Values")
        st.dataframe(df.isna().sum(), use_container_width=True)
    with col2:
        st.subheader("Row Count")
        st.metric("Rows", len(df))

    if st.button("✨ Normalize text + domains"):
        show_loader("Cleaning text & domains...")
        tcol = find_col(df, ["text"])
        dcol = find_col(df, ["domain"])
        if tcol:
            df[tcol] = df[tcol].astype(str).str.strip()
        if dcol:
            df[dcol] = df[dcol].astype(str).str.strip().str.lower()
        st.session_state["uploaded_df"] = df
        st.success("Preprocessing applied.")


def page_ner():
    themed_header("Heuristic Entity Extraction", "🧠")
    df = st.session_state.get("uploaded_df")
    if df is None:
        st.info("Upload a dataset first.")
        return

    text_col = find_col(df, ["text"])
    if not text_col:
        st.error("No `text` column found in dataset.")
        return

    if st.button("▶ Run NER (heuristic)"):
        show_loader("Extracting entities...")
        rows = []
        for _, r in df.iterrows():
            txt = str(r[text_col])
            ents = extract_entities_heuristic(txt)
            rows.append({
                "domain": r.get(find_col(df, ["domain"]), ""),
                "text": txt,
                "entities": str(ents),
            })
        ner_df = pd.DataFrame(rows)
        st.session_state["ner_df"] = ner_df
        ner_df.to_csv(ENTITIES_OUT, index=False)
        st.success(f"Entities extracted and saved → {ENTITIES_OUT}")

    ner_df = st.session_state.get("ner_df")
    if ner_df is not None:
        st.subheader("Entities (sample)")
        st.dataframe(ner_df.head(50), use_container_width=True)


def page_relations():
    themed_header("Relation Extraction (Heuristic S-V-O)", "🔗")
    df = st.session_state.get("uploaded_df")
    if df is None:
        st.info("Upload a dataset first.")
        return

    text_col = find_col(df, ["text"])
    if not text_col:
        st.error("No `text` column found in dataset.")
        return

    if st.button("▶ Run Relation Extraction"):
        show_loader("Mining subject-verb-object triples...")
        rows = []
        for _, r in df.iterrows():
            txt = str(r[text_col])
            triples = extract_triples_heuristic(txt)
            for s, rel, o in triples:
                rows.append({
                    "text": txt,
                    "subject": s,
                    "relation": rel,
                    "object": o,
                })
        rel_df = pd.DataFrame(rows)
        st.session_state["rel_df"] = rel_df
        rel_df.to_csv(RELATIONS_OUT, index=False)
        st.success(f"Relations extracted and saved → {RELATIONS_OUT}")

    rel_df = st.session_state.get("rel_df")
    if rel_df is not None:
        st.subheader("Relations (sample)")
        st.dataframe(rel_df.head(50), use_container_width=True)

# ------------------------------------------------------------------------------
# TABLE PAGES
# ------------------------------------------------------------------------------

def page_entities_table():
    themed_header("Entities Table", "📋")
    ner_df = st.session_state.get("ner_df")
    if ner_df is None:
        st.info("Run NER extraction first.")
        return
    st.dataframe(ner_df, use_container_width=True)


def page_relations_table():
    themed_header("Relations Table", "📚")
    rel_df = st.session_state.get("rel_df")
    if rel_df is None:
        st.info("Run relation extraction first.")
        return
    st.dataframe(rel_df, use_container_width=True)

# ------------------------------------------------------------------------------
# FULL GRAPH PAGE
# ------------------------------------------------------------------------------

def page_full_graph():
    themed_header("Full Knowledge Graph", "🌐")
    rel_df = st.session_state.get("rel_df")
    if rel_df is None:
        st.info("Run relation extraction first.")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("⚙ Build & Save Interactive Graph"):
            show_loader("Building graph with PyVis...")
            G = build_graph_from_triples_df(rel_df)
            path = save_pyvis_graph(G, GRAPH_HTML)
            st.session_state["last_full_graph_html"] = path
            st.success(f"Graph saved → {path}")
    with col2:
        st.write("After building, the graph will appear below and is also available as an HTML file.")

    if "last_full_graph_html" in st.session_state and os.path.exists(
        st.session_state["last_full_graph_html"]
    ):
        html_path = st.session_state["last_full_graph_html"]
        st.components.v1.html(
            open(html_path, "r", encoding="utf-8").read(),
            height=650,
        )
    else:
        st.info("Build the graph first to view it here.")

# ------------------------------------------------------------------------------
# SEMANTIC SEARCH + SUBGRAPH
# ------------------------------------------------------------------------------

def page_search():
    themed_header("Semantic Search & Subgraph Generator", "🔍")
    rel_df = st.session_state.get("rel_df")
    df = st.session_state.get("uploaded_df")
    if rel_df is None or df is None:
        st.info("Make sure dataset is uploaded and relations extracted.")
        return

    text_col = find_col(df, ["text"])
    if not text_col:
        st.error("No `text` column found.")
        return

    sentences = df[text_col].astype(str).tolist()

    subj_c = find_col(rel_df, ["subject"])
    rel_c = find_col(rel_df, ["relation", "verb"])
    obj_c = find_col(rel_df, ["object"])
    if not subj_c or not rel_c or not obj_c:
        st.error("Relations table must have subject / relation / object columns.")
        return

    relation_strings = rel_df.apply(
        lambda r: f"{r[subj_c]} {r[rel_c]} {r[obj_c]}",
        axis=1,
    ).tolist()

    vec, sent_vecs, rel_vecs, _ = build_vectorizers(sentences, relation_strings)
    if vec is None:
        st.error("Not enough data to build TF-IDF vectors.")
        return

    query = st.text_input(
        "Search for concepts (e.g., 'Who launched a program?', 'Finance reports', 'AI collaborations')"
    )
    n_hits = st.slider("Top results", 1, 10, 3)

    if st.button("🔎 Search & Build Subgraph"):
        if not query.strip():
            st.warning("Enter a query first.")
            return

        show_loader("Running semantic search...")

        qv = vec.transform([query])
        sent_scores = linear_kernel(qv, sent_vecs).flatten()
        rel_scores = linear_kernel(qv, rel_vecs).flatten()

        top_sent_idx = sent_scores.argsort()[::-1][:n_hits]
        top_rel_idx = rel_scores.argsort()[::-1][:n_hits]

        st.subheader("📌 Top Matching Sentences")
        for i in top_sent_idx:
            st.write(f"- {sentences[i]}  *(score {sent_scores[i]:.3f})*")

        matched_rels = rel_df.iloc[top_rel_idx]

        st.subheader("🔗 Triples Used for Subgraph")
        st.dataframe(matched_rels, use_container_width=True)

        # Build & render subgraph
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
        net = Network(height="520px", width="100%", directed=True, bgcolor="#020308")
        net.toggle_physics(True)

        for n in G_sub.nodes():
            net.add_node(n, label=n, color="#00eaff", size=20)

        for u, v, data in G_sub.edges(data=True):
            label = data.get("label", "")
            net.add_edge(u, v, title=label, label=label, color="#ff00d4")

        net.save_graph(sub_html)
        st.components.v1.html(open(sub_html, "r", encoding="utf-8").read(), height=520)

        # log search for Subgraph Viewer
        st.session_state.setdefault("query_log", []).append(
            {"query": query, "top_rel_idx": top_rel_idx.tolist()}
        )

# ------------------------------------------------------------------------------
# SUBGRAPH VIEWER
# ------------------------------------------------------------------------------

def page_subgraph_viewer():
    themed_header("Previous Semantic Subgraphs", "🕸️")
    logs = st.session_state.get("query_log", [])
    rel_df = st.session_state.get("rel_df")
    if not logs or rel_df is None:
        st.info("Run at least one semantic search to see saved subgraphs.")
        return

    choice = st.selectbox(
        "Select a previous query",
        list(range(len(logs))),
        format_func=lambda i: f"{i+1}. {logs[i]['query']}",
    )
    entry = logs[choice]
    idxs = entry.get("top_rel_idx", [])
    matched = rel_df.iloc[idxs]

    st.subheader("Relations for this query")
    st.dataframe(matched, use_container_width=True)

    if st.button("Visualize subgraph"):
        s_col = find_col(rel_df, ["subject"])
        r_col = find_col(rel_df, ["relation", "verb"])
        o_col = find_col(rel_df, ["object"])
        G_sub = nx.DiGraph()
        for _, r in matched.iterrows():
            s = str(r[s_col]).strip()
            o = str(r[o_col]).strip()
            rel = str(r[r_col]).strip()
            if s and o:
                G_sub.add_node(s)
                G_sub.add_node(o)
                G_sub.add_edge(s, o, label=rel)

        sub_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
        net = Network(height="520px", width="100%", directed=True, bgcolor="#020308")
        net.toggle_physics(True)
        for n in G_sub.nodes():
            net.add_node(n, label=n, color="#00eaff", size=18)
        for u, v, data in G_sub.edges(data=True):
            label = data.get("label", "")
            net.add_edge(u, v, title=label, label=label, color="#ff00d4")
        net.save_graph(sub_html)
        st.components.v1.html(open(sub_html, "r", encoding="utf-8").read(), height=520)

# ------------------------------------------------------------------------------
# ADMIN DASHBOARD
# ------------------------------------------------------------------------------

def page_admin():
    themed_header("Admin Dashboard & Feedback", "📊")
    rel_df = st.session_state.get("rel_df")
    ner_df = st.session_state.get("ner_df")
    df = st.session_state.get("uploaded_df")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entities", len(ner_df) if ner_df is not None else 0)
    c2.metric("Relations", len(rel_df) if rel_df is not None else 0)

    domain_count = 0
    if df is not None:
        dcol = find_col(df, ["domain"])
        if dcol:
            domain_count = df[dcol].nunique()
    c3.metric("Domains", domain_count)

    searches = len(st.session_state.get("query_log", []))
    c4.metric("Semantic Queries", searches)

    st.markdown("### Relation Distribution")
    if rel_df is not None:
        rcol = find_col(rel_df, ["relation", "verb"])
        if rcol:
            st.bar_chart(rel_df[rcol].value_counts().head(10))
        else:
            st.info("No relation column found.")
    else:
        st.info("No relations yet.")

    st.markdown("### Feedback & Logs")
    logs = st.session_state.get("app_logs", [])
    if logs:
        st.dataframe(pd.DataFrame(logs[-40:]), use_container_width=True)
    else:
        st.info("Logs will appear here as you use the app.")

    st.markdown("### Graph Cleaning")
    if clean_graph is None:
        st.info(
            "Add a `modules/graph_cleaner.py` file with a `clean_graph(df)` function "
            "to enable automated cleaning (duplicate merge, generic node removal, etc.)."
        )
    else:
        if st.button("🧹 Run Graph Cleaning"):
            if rel_df is None:
                st.warning("No relations to clean.")
            else:
                try:
                    show_loader("Cleaning graph relations...")
                    cleaned = clean_graph(rel_df.copy())
                    st.session_state["rel_df"] = cleaned
                    cleaned.to_csv(PROCESSED_DIR / "relations_cleaned.csv", index=False)
                    st.success(
                        "Graph cleaned and saved → data/processed/relations_cleaned.csv"
                    )
                except Exception as e:
                    st.error("Cleaning failed: " + str(e))

# ------------------------------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------------------------------

PAGES = [
    ("Welcome", page_welcome),
    ("Upload Dataset", page_upload),
    ("Preprocessing", page_preprocessing),
    ("NER Extraction", page_ner),
    ("Relation Extraction", page_relations),
    ("Entities Table", page_entities_table),
    ("Relations Table", page_relations_table),
    ("Full Knowledge Graph", page_full_graph),
    ("Semantic Search", page_search),
    ("Subgraph Viewer", page_subgraph_viewer),
    ("Admin Dashboard", page_admin),
]

# Session initial
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

st.sidebar.title("AI-KnowMap")
st.sidebar.write("---")

# If not logged in → only show auth
if not st.session_state["logged_in"]:
    page_auth()
    st.stop()

# Sidebar UI after login
st.sidebar.markdown(
    """
    <div style="padding-top:0.75rem;padding-bottom:0.4rem;">
      <div class="sidebar-logo">AI-KnowMap</div>
      <div class="sidebar-sub">Cross-Domain Knowledge Graph Studio</div>
    </div>
    """,
    unsafe_allow_html=True,
)

menu = st.sidebar.radio("Menu", [p[0] for p in PAGES])

if st.sidebar.button("Logout ❌"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# Log visits
st.session_state.setdefault("app_logs", []).append(
    {"time": pd.Timestamp.now(), "event": f"Visited {menu}"}
)

# Dispatch selected page
for name, func in PAGES:
    if name == menu:
        func()
        break

st.markdown("---")
st.caption("AI-KnowMap — Neon Knowledge Graph Studio (no spaCy).")
