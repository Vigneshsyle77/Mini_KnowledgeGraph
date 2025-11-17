# modules/graph_cleaner.py
import pandas as pd
import networkx as nx


# --------------------------------------------------------
# 1. Normalize text (remove case differences, whitespace)
# --------------------------------------------------------
def normalize(text):
    if not isinstance(text, str):
        return text
    return text.strip().lower()


# --------------------------------------------------------
# 2. Detect Duplicate Nodes (Google/google, Harvard/harvard)
# --------------------------------------------------------
def detect_duplicate_nodes(df_rel):
    subj_norm = df_rel["subject"].apply(normalize)
    obj_norm = df_rel["object"].apply(normalize)

    all_nodes = pd.concat([subj_norm, obj_norm]).tolist()

    mapping = {}
    
    for node in set(all_nodes):
        # normalized → original
        originals = [n for n in all_nodes if normalize(n) == node]
        representative = originals[0]  # choose first as canonical
        for o in originals:
            mapping[o] = representative

    return mapping


# --------------------------------------------------------
# 3. Apply duplicate merge
# --------------------------------------------------------
def merge_duplicates(df_rel, mapping):
    df_rel["subject"] = df_rel["subject"].map(lambda x: mapping.get(x, x))
    df_rel["object"] = df_rel["object"].map(lambda x: mapping.get(x, x))
    return df_rel


# --------------------------------------------------------
# 4. Remove Low-value / Noise Nodes
# --------------------------------------------------------
NOISE_NODES = {"2023", "2024", "10%", "last", "quarter", "study"}

def remove_noise_relations(df_rel):
    return df_rel[
        ~df_rel["subject"].astype(str).str.lower().isin(NOISE_NODES)
        &
        ~df_rel["object"].astype(str).str.lower().isin(NOISE_NODES)
    ]


# --------------------------------------------------------
# 5. Remove orphan nodes (single node with zero relations)
# --------------------------------------------------------
def remove_orphan_nodes(df_rel):
    all_nodes = set(df_rel["subject"]).union(set(df_rel["object"]))

    used_nodes = set()
    for _, row in df_rel.iterrows():
        used_nodes.add(row["subject"])
        used_nodes.add(row["object"])

    valid_nodes = all_nodes.intersection(used_nodes)

    return df_rel[
        df_rel["subject"].isin(valid_nodes) &
        df_rel["object"].isin(valid_nodes)
    ]


# --------------------------------------------------------
# 6. Fix common incorrect triples
# --------------------------------------------------------
FIX_RULES = {
    "launched": ["June", "2023", "2024"],
    "reported": ["quarter"],
    "rise": ["revenue"]
}

def fix_incorrect_relations(df_rel):
    cleaned = []
    for _, r in df_rel.iterrows():
        subj = r["subject"]
        rel = r["relation"]
        obj = r["object"]

        # If object is wrong according to fix rules → skip
        if rel in FIX_RULES and obj in FIX_RULES[rel]:
            continue  

        cleaned.append(r)

    return pd.DataFrame(cleaned)


# --------------------------------------------------------
# 7. Run full cleaning pipeline
# --------------------------------------------------------
def clean_graph(df_rel):
    print("🔧 Cleaning Graph...")

    # Step 1: Fix wrong triples
    df_rel = fix_incorrect_relations(df_rel)

    # Step 2: Remove noise nodes
    df_rel = remove_noise_relations(df_rel)

    # Step 3: Detect and merge duplicates
    mapping = detect_duplicate_nodes(df_rel)
    df_rel = merge_duplicates(df_rel, mapping)

    # Step 4: Remove orphans
    df_rel = remove_orphan_nodes(df_rel)

    # Reset index
    df_rel = df_rel.reset_index(drop=True)

    return df_rel
