# 🧩 Mini Knowledge Graph Builder (AI-KnowMap)

## 📘 Overview

The **Mini Knowledge Graph Builder (AI-KnowMap)** is a Python + Streamlit project that extracts entities and relationships from text data and visualizes them as an interactive **Knowledge Graph**.  
It demonstrates core concepts of:

- Natural Language Processing (NLP)
- Lightweight Named Entity Recognition (NER)
- Relation Extraction (Subject–Relation–Object)
- Knowledge Graph Construction
- Semantic Search & Subgraph Generation
- Admin Dashboard & Feedback View
- Containerization & Cloud Deployment (Railway)

---

## 📂 Project Structure

```bash
Mini_KnowledgeGraph/
│
├── data/
│   ├── cross_domain_dataset.csv          # Main input dataset (domain, text)
│   ├── cross_domain_dataset_backup.csv   # Backup of original dataset
│   ├── entities_out.csv                  # Extracted entities (heuristic NER)
│   ├── relations_out.csv                 # Extracted relations (S–R–O)
│   └── processed/                        # Saved copies & cleaned data
│
├── modules/
│   ├── app.py                            # Main Streamlit app (AI-KnowMap Studio)
│   ├── graph_cleaner.py                  # (Optional) graph cleaning helpers
│   ├── ner_extraction.py                 # Legacy script for NER (spaCy version, if used)
│   ├── relation_extraction.py            # Legacy relation extractor (script version)
│   ├── graph_builder.py                  # Legacy graph builder (script version)
│   └── semantic_search.py                # Legacy semantic search script
│
├── ui/
│   ├── knowledge_graph.html              # Full interactive PyVis graph
│   ├── Screenshot1.png                   # Sample graph screenshot
│   └── Screenshot2.png                   # Sample graph screenshot
│
├── Dockerfile                            # Container image definition
├── requirements.txt                      # Python dependencies
├── README.md                             # Project documentation (this file)
└── Mini_KnowledgeGraph_Report.docx       # Milestone report
```

> Note: Some legacy modules (like `ner_extraction.py`) were used in earlier milestones.  
> The current main entrypoint is **`modules/app.py`** which integrates everything inside a unified Streamlit UI.

---

## ✨ Core Features

- 📁 **Upload Any CSV Dataset** with at least a `text` column (and optionally `domain`).
- 🧠 **Heuristic NER** (no spaCy required) to detect ORG, PERSON, DATE, PERCENT etc.
- 🔗 **Relation Extraction** using verb-based rules and NER fallbacks.
- 🌐 **Knowledge Graph Visualization** using NetworkX + PyVis.
- 🔍 **Semantic Search** using TF‑IDF + cosine similarity.
- 🕸️ **Query-Based Subgraph Generation** (only triples related to your query).
- 📊 **Admin Dashboard**: metrics, relation distribution, basic logs, optional cleaning.
- 🧪 **Dockerized & Cloud Deployed** on **Railway**.

---

## ⚙️ Local Setup Instructions (Without Docker)

### 1️⃣ Create and Activate a Virtual Environment

```bash
# From project root (Mini_KnowledgeGraph/)
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2️⃣ Install Dependencies

> The modern app uses **heuristic NER** and no longer requires spaCy.  
> Just install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
# From project root
streamlit run modules/app.py
```

Then open in your browser:

```text
http://localhost:8501
```

You will see the **AI-KnowMap Studio** neon UI with sidebar navigation:
- Login / Signup
- Upload Dataset
- Preprocessing
- NER Extraction
- Relation Extraction
- Entities Table
- Relations Table
- Full Knowledge Graph
- Semantic Search
- Subgraph Viewer
- Admin Dashboard

---

## 🧠 How the Pipeline Works

1. **Upload Dataset**
   - Upload a CSV with columns like `domain` and `text` (or use the sample dataset).
   - Data is stored in memory and optionally saved to `data/processed/cross_domain_dataset.csv`.

2. **Preprocessing**
   - Strips whitespace and normalizes `domain` values to lowercase.
   - Shows basic data health (missing values, row count).

3. **Heuristic NER**
   - Uses regex + keyword heuristics (no spaCy) to detect:
     - `PERSON` (titles and double-capital names)
     - `ORG` (keywords like *University, Google, NASA*)
     - `DATE` (years, month–year patterns)
     - `PERCENT` (e.g. `10%`)
   - Output format (stored in `entities_out.csv`):
     ```text
     domain, text, entities
     Finance, "Goldman Sachs reported a 10% rise...", "[('Goldman Sachs','ORG'), ('10%','PERCENT'), ...]"
     ```

4. **Relation Extraction**
   - Uses verb keywords (`launched`, `reported`, `partnered`, etc.).
   - Extracts **Subject – Relation – Object** triples with a fallback using NER entities.
   - Output format (stored in `relations_out.csv`):
     ```text
     text, subject, relation, object
     "Harvard University launched a new AI program...", "Harvard University", "launched", "AI program"
     ```

5. **Knowledge Graph Creation**
   - Builds a directed graph with **NetworkX**:
     - Nodes → subjects and objects
     - Edges → labelled with the `relation`
   - Rendered as an interactive PyVis HTML file:
     ```text
     ui/knowledge_graph.html
     ```

6. **Semantic Search & Subgraphs**
   - Uses `TfidfVectorizer` + cosine similarity on:
     - Raw sentences (`text` column)
     - Relation strings (`subject relation object`)
   - Given a query like *"Who launched a program?"* the app:
     - Ranks similar sentences
     - Ranks similar triples
     - Builds a **focused subgraph** only from those triples

7. **Admin Dashboard**
   - Shows:
     - Total entities
     - Total relations
     - Unique domains
     - Relation frequency chart
     - Basic logs (visits, semantic search runs)
   - Optionally integrates a `graph_cleaner.clean_graph(df)` function when present to:
     - Remove generic nodes
     - Merge duplicates
     - Drop low-value edges

---

## 🔍 Example Semantic Search

### Query

```text
Who founded a company?
```

### Example Output

```text
✅ Elon Musk founded SpaceX. (score: 0.84)
✅ Jeff Bezos founded Amazon. (score: 0.82)
✅ Steve Jobs established Apple. (score: 0.79)
```

### Query

```text
Science discoveries
```

### Example Output

```text
✅ Marie Curie discovered Radium. (score: 0.83)
✅ Isaac Newton formulated the Laws of Motion. (score: 0.81)
✅ Albert Einstein developed the Theory of Relativity. (score: 0.80)
```

---

## 📊 Sample Graph Output

Below are example screenshots from the generated graph (paths may vary):

| Visualization 1 | Visualization 2 |
|-----------------|-----------------|
| ![Graph 1](ui/Screenshot1.png) | ![Graph 2](ui/Screenshot2.png) |

---

## 🐳 Docker Usage (Local)

> The app is containerized so you can run it without installing Python locally.

### 1️⃣ Build the Image

From the project root:

```bash
docker build -t ai-knowmap .
```

### 2️⃣ Run the Container

```bash
docker run -p 8501:8501 ai-knowmap
```

Then open:

```text
http://localhost:8501
```

You should see the AI-KnowMap UI running from inside the Docker container.

---

## ☁️ Cloud Deployment (Railway)

This project is deployed on **Railway** at:

```text
https://miniknowledgegraph-production.up.railway.app
```

### Railway Deployment Summary

- **Platform:** Railway (PaaS)
- **Runtime:** Dockerfile-based deployment
- **App Type:** Streamlit web app
- **Main entrypoint:** `modules/app.py`

#### Steps Followed

1. Pushed the project (with `Dockerfile` and `requirements.txt`) to GitHub.
2. Created a new project on **Railway** and linked the GitHub repository.
3. Let Railway automatically build the Docker image using the Dockerfile.
4. Exposed port `8501` (Streamlit default) in the Dockerfile and Railway settings.
5. Deployed and verified the app using the generated Railway URL.
6. Tested:
   - Dataset upload
   - NER extraction
   - Relation extraction
   - Full graph visualization
   - Semantic search & subgraph

#### What Worked Well

- Automatic builds from GitHub commits.
- Simple exposure of the Streamlit port via Docker.
- Easy rollback and redeploy from the Railway dashboard.

#### Issues Faced & Fixes

- **Issue:** Large dependencies (NLP & graph libs) leading to longer build times.  
  **Fix:** Removed unused heavy dependencies from `requirements.txt` and removed spaCy from the production image.

- **Issue:** Path confusion for `modules/app.py` inside Docker.  
  **Fix:** Explicitly set the working directory and used `streamlit run modules/app.py` as the `CMD` in Dockerfile.

#### “How to Deploy” (Railway)

1. Fork or push this repo into your own GitHub.
2. Create a new project on Railway, choose “Deploy from GitHub”.  
3. Select this repository and confirm.
4. Ensure your **Dockerfile** is present at the repo root.
5. Railway will:
   - Build Docker image
   - Install Python dependencies
   - Start the Streamlit server
6. Once deployment succeeds, open the public URL provided by Railway.

You can add this section to your own README under **Deployment** if you fork or extend the project.

---

## 🧪 Testing & Optimization Notes

During deployment and testing, the following checks were performed:

- ✅ Upload dataset and preview table.
- ✅ Run heuristic NER and verify `entities_out.csv`.
- ✅ Run relation extraction and inspect `relations_out.csv`.
- ✅ Build & open the full PyVis knowledge graph.
- ✅ Perform semantic queries and verify meaningful subgraphs.

### Optimizations Applied

- Removed spaCy + large transformer models to keep the Docker image smaller and builds faster.
- Switched to **heuristic NER** with regex and simple rules (good enough for milestone scope).
- Reduced logging noise in Streamlit app.
- Cached some dataset operations in `st.session_state` to avoid recomputation on each page.

Future improvements could include:

- Real database-backed logging (PostgreSQL / MongoDB).
- More advanced NER / relation extraction models (if compute allows).
- Role-based admin vs. user dashboards.
- Export of cleaned graph in Neo4j / RDF formats.

---

## 🧾 Example Triples

| Subject        | Relation    | Object      |
|----------------|------------|------------|
| University     | launched   | June       |
| Johnson        | conducted  | 2023       |
| Google         | partnered  | NASA       |
| Goldman Sachs  | reported   | 10%        |

---

## 💡 Reflection

> Through this project, I learned how to extract meaningful structure from unstructured text data and represent it visually using a Knowledge Graph.  
> I also understood how different modules—NER, relation extraction, semantic search, and visualization—can be integrated into a single interactive web app, and how to containerize and deploy it to the cloud.

---

## 👨‍💻 Author

**Saivignesh Marapelli**  
📅 *Milestone 1 & 2 – AI-KnowMap Project*  
📍 Built with Python, Streamlit, NetworkX, PyVis, scikit-learn, and Docker.
