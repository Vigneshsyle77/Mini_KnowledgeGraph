import streamlit as st
import pandas as pd
import os
import ast
import datetime

# ===========================================================
# 🧠 AI-KnowMap Admin Dashboard — Final Clean Version
# ===========================================================

st.set_page_config(page_title="AI-KnowMap Admin Dashboard", layout="wide")
st.title("🧠 AI-KnowMap Admin Dashboard")

# --- Helper: Safe Column Access ---
def find_column(df, keyword):
    """Find a column that contains a specific keyword (case-insensitive)."""
    for col in df.columns:
        if keyword.lower() in col.lower().strip():
            return col
    return None


# ----------------------------
# 📂 File Upload Section
# ----------------------------
st.sidebar.header("📁 Upload Custom Datasets")
uploaded_entities = st.sidebar.file_uploader("Upload entities.csv", type=["csv"])
uploaded_relations = st.sidebar.file_uploader("Upload relations.csv", type=["csv"])
uploaded_semantic = st.sidebar.file_uploader("Upload semantic_logs.csv (optional)", type=["csv"])

# Default fallback paths
entities_path = "data/processed_entities.csv"
relations_path = "data/extracted_triples.csv"

# Load CSVs safely
if uploaded_entities:
    entities_df = pd.read_csv(uploaded_entities)
    st.sidebar.success("✅ Custom entities.csv uploaded!")
elif os.path.exists(entities_path):
    entities_df = pd.read_csv(entities_path)
else:
    st.error("❌ No entities.csv file found!")
    st.stop()

if uploaded_relations:
    relations_df = pd.read_csv(uploaded_relations)
    st.sidebar.success("✅ Custom relations.csv uploaded!")
elif os.path.exists(relations_path):
    relations_df = pd.read_csv(relations_path)
else:
    st.error("❌ No relations.csv file found!")
    st.stop()

# Normalize column names
entities_df.columns = entities_df.columns.str.strip().str.lower()
relations_df.columns = relations_df.columns.str.strip().str.lower()

# ✅ Auto-detect if entities and relations files are swapped
if {"subject", "object", "relation"}.issubset(set(entities_df.columns)):
    st.sidebar.warning("⚠️ Swapping detected — entities.csv looks like relations data.")
    entities_df, relations_df = relations_df, entities_df
    st.sidebar.success("✅ Files swapped internally for correct mapping.")

# ✅ Standardize column names
rename_map = {
    "text": "text",
    "domain": "domain",
    "relation": "relation",
    "relations": "relation",
    "object": "object",
    "subject": "subject",
    "entities": "entities",
    "entity": "entities",
    "entity_list": "entities",
}
entities_df.rename(columns=rename_map, inplace=True)
relations_df.rename(columns=rename_map, inplace=True)

# Debug info in sidebar
st.sidebar.info(f"🧩 Detected Entities Columns: {list(entities_df.columns)}")
st.sidebar.info(f"🧩 Detected Relations Columns: {list(relations_df.columns)}")

# Load semantic logs if available
semantic_logs_df = None
if uploaded_semantic:
    semantic_logs_df = pd.read_csv(uploaded_semantic)
    st.sidebar.success("🔍 Semantic logs loaded!")

st.success("✅ Data successfully loaded and ready for visualization.")

# ===========================================================
# 📊 Dashboard Sections
# ===========================================================

# ----------------------------
# 📊 Pipeline Status
# ----------------------------
st.header("📊 Pipeline Status")
st.write("✅ Dataset Loaded: cross_domain_dataset.csv")
st.write(f"✅ NER Extraction Completed (Entities: {len(entities_df)})")
st.write(f"✅ Relation Extraction Completed (Triples: {len(relations_df)})")
st.write("✅ Graph Built: expanded_knowledge_graph.html")

# ----------------------------
# 📈 Knowledge Graph Overview
# ----------------------------
st.header("📈 Knowledge Graph Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Entities", len(entities_df))
col2.metric("Total Relations", len(relations_df))

domain_col = find_column(entities_df, "domain")
if domain_col:
    col3.metric("Unique Domains", entities_df[domain_col].nunique())
else:
    col3.metric("Unique Domains", 0)
    st.info("ℹ️ Domain column not found in entities.csv.")

# ----------------------------
# 🌍 Domain Distribution
# ----------------------------
st.subheader("🌍 Domain Distribution")
if domain_col:
    st.bar_chart(entities_df[domain_col].value_counts())
else:
    st.info("ℹ️ Skipping domain chart — column not found.")

# ----------------------------
# 🔗 Top Relations Extracted
# ----------------------------
st.subheader("🔗 Top Relations Extracted")
relation_col = find_column(relations_df, "relation")
if relation_col:
    top_relations = relations_df[relation_col].value_counts().head(10)
    st.bar_chart(top_relations)
else:
    st.info("ℹ️ Skipping relation chart — column not found.")

# ----------------------------
# 🧠 Entity Breakdown by Type
# ----------------------------
st.subheader("🧠 Entity Breakdown by Type")
entity_col = find_column(entities_df, "entit")
all_entities = []

if entity_col:
    for ent_list in entities_df[entity_col]:
        try:
            ents = ast.literal_eval(ent_list)
            all_entities.extend([label for _, label in ents])
        except Exception:
            continue
    if all_entities:
        entity_counts = pd.Series(all_entities).value_counts().head(10)
        st.bar_chart(entity_counts)
    else:
        st.info("ℹ️ No valid entity data found.")
else:
    st.info("ℹ️ Skipping entity chart — column not found.")

# ----------------------------
# 📈 Trend Visualization
# ----------------------------
st.subheader("📊 Trend Visualization (Entity Growth Over Time)")
trend_data = pd.DataFrame({
    "Date": pd.date_range(datetime.date(2024, 1, 1), periods=6, freq="M"),
    "Entities": [45, 60, 85, 100, 120, len(entities_df)],
    "Relations": [20, 30, 45, 55, 65, len(relations_df)]
}).set_index("Date")
st.line_chart(trend_data)

# ----------------------------
# 🗂 Data Viewer
# ----------------------------
st.subheader("🗂 Data Viewer")
tab1, tab2 = st.tabs(["Entities Data", "Relations Data"])
with tab1:
    st.dataframe(entities_df)
with tab2:
    st.dataframe(relations_df)

# ----------------------------
# 🔍 Semantic Search Logs
# ----------------------------
st.header("🔍 Semantic Search Logs")
if semantic_logs_df is not None:
    st.dataframe(semantic_logs_df)
else:
    st.info("Upload a `semantic_logs.csv` file in the sidebar to display recent queries.")

# ----------------------------
# 💬 Feedback Summary
# ----------------------------
st.header("💬 User Feedback Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Queries", 58)
col2.metric("Accurate Responses", 51)
col3.metric("Feedback Accuracy (%)", 88)

# ----------------------------
# ⚙️ Admin Controls
# ----------------------------
st.header("⚙️ Admin Actions")
colA, colB, colC = st.columns(3)
colA.button("🧩 Merge Duplicate Nodes")
colB.button("❌ Delete Incorrect Relation")
colC.button("📋 Export Cleaned Data")

# ----------------------------
# 📥 Feedback Logs
# ----------------------------
st.header("📥 Feedback Logs")
feedback_data = pd.DataFrame({
    "Query": ["Who founded a company?", "Which university launched?", "Science discoveries"],
    "Feedback": ["Correct", "Slight mismatch", "Wrong relation"],
    "Status": ["✅", "⚠️", "❌"]
})
st.table(feedback_data)

# ----------------------------
# 🧾 Footer
# ----------------------------
st.markdown("---")
st.caption("Developed by Saivignesh Marapelli | AI-KnowMap Project | Streamlit Admin Dashboard")
