# modules/graph_cleaner.py
import pandas as pd
import networkx as nx

def normalize(text):
    if not isinstance(text, str):
        return text
    return text.strip().lower()

NOISE_NODES = {"2023", "2024", "10%", "last", "quarter", "study", "percent", ""}

def detect_duplicate_nodes(df_rel):
    subj_norm = df_rel["subject"].astype(str).apply(normalize)
    obj_norm = df_rel["object"].astype(str).apply(normalize)

    node_map = {}
    # Build canonical representative mapping (normalized -> representative)
    for orig, norm in pd.concat([df_rel["subject"], df_rel["object"]]).astype(str).map(lambda x: (x, normalize(x))):
        pass  # not used, we iterate differently below

    # Collect all originals for each normalized form
    all_nodes = list(df_rel["subject"].astype(str)) + list(df_rel["object"].astype(str))
    norm_to_originals = {}
    for n in all_nodes:
        key = normalize(n)
        norm_to_originals.setdefault(key, []).append(n)

    mapping = {}
    for key, originals in norm_to_originals.items():
        # Choose most frequent original as representative (or first)
        rep = max(originals, key=lambda x: originals.count(x))
        for o in originals:
            mapping[o] = rep

    return mapping

def merge_duplicates(df_rel, mapping):
    df_rel["subject"] = df_rel["subject"].map(lambda x: mapping.get(str(x), x))
    df_rel["object"]  = df_rel["object"].map(lambda x: mapping.get(str(x), x))
    return df_rel

def remove_noise_relations(df_rel):
    low = set([s.lower() for s in NOISE_NODES])
    mask = (~df_rel["subject"].astype(str).str.lower().isin(low)) & (~df_rel["object"].astype(str).str.lower().isin(low))
    return df_rel[mask]

FIX_RULES = {
    "launched": {"june", "october", "july"},
    "reported": {"quarter"},
    "released": {"october"},
}

def fix_incorrect_relations(df_rel):
    cleaned_rows = []
    for _, r in df_rel.iterrows():
        subj = str(r["subject"])
        rel  = str(r["relation"]).lower()
        obj  = str(r["object"])
        # if object mistakenly is a date token in FIX_RULES, try to skip
        bad_objs = FIX_RULES.get(rel, set())
        if obj.lower() in bad_objs:
            # don't keep the triple as-is; skip (will be refined manually)
            continue
        cleaned_rows.append(r)
    return pd.DataFrame(cleaned_rows)

def remove_orphan_nodes(df_rel):
    nodes = set(df_rel["subject"]).union(set(df_rel["object"]))
    used = set()
    for _, r in df_rel.iterrows():
        used.add(r["subject"])
        used.add(r["object"])
    valid = nodes.intersection(used)
    return df_rel[df_rel["subject"].isin(valid) & df_rel["object"].isin(valid)]

def clean_graph(df_rel):
    """
    Expects DataFrame with columns: subject, relation, object (case-insensitive names OK)
    Returns cleaned DataFrame (reset index).
    """
    # normalize columns (lowercase names)
    cols = {c.lower(): c for c in df_rel.columns}
    # try to rename to expected names if they differ
    for expected in ("subject", "relation", "object"):
        if expected not in cols:
            # try to find similar column (case-insensitive)
            for c in df_rel.columns:
                if c.strip().lower() == expected:
                    cols[expected] = c
                    break
    # If still not found, raise
    if "subject" not in cols or "relation" not in cols or "object" not in cols:
        raise ValueError("relations DataFrame must contain subject, relation, object columns (case-insensitive). Detected columns: %s" % list(df_rel.columns))

    df_rel = df_rel.rename(columns={cols["subject"]: "subject", cols["relation"]: "relation", cols["object"]: "object"})
    # run fixes
    df_rel = fix_incorrect_relations(df_rel)
    df_rel = remove_noise_relations(df_rel)
    mapping = detect_duplicate_nodes(df_rel)
    df_rel = merge_duplicates(df_rel, mapping)
    df_rel = remove_orphan_nodes(df_rel)
    df_rel = df_rel.reset_index(drop=True)
    return df_rel
