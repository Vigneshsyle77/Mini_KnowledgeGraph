# modules/graph_cleaner.py
"""
Graph cleaner scaffold for Mini_KnowMap.
Provides a clean_graph(df_relations: pd.DataFrame) -> pd.DataFrame function.

What it does (safe defaults):
- Detects columns subject/relation/object (case-insensitive)
- Normalizes entity strings (strip, unify spaces, remove surrounding punctuation)
- Merges exact duplicates (case-insensitive)
- Removes generic/low-value nodes and edges
- Drops orphan triples (missing subject/object) and self-loops
- Standardizes relation verbs (simple synonyms map)
- Returns cleaned DataFrame with columns: subject, relation, object (original casing preserved where possible)
"""

import re
import unicodedata
import pandas as pd

# Config: customize lists to your domain
GENERIC_ENTITIES = set([
    "study", "research", "project", "annual", "increase", "report", "reports", "data", "information",
    "cases", "case", "results", "findings"
])

RELATION_SYNONYMS = {
    "reports": "reported",
    "report": "reported",
    "announces": "announced",
    "announce": "announced",
    "partners": "partnered",
    "collaborates": "partnered",
    "collaborated": "partnered",
    "releases": "released",
    "release": "released",
    "launched": "launched",
    "launches": "launched",
    "invests": "invested"
}

def _normalize_text(s):
    """Return a normalized string used for deduping/matching."""
    if pd.isna(s):
        return ""
    s = str(s).strip()
    # remove surrounding punctuation
    s = re.sub(r'^[\W_]+|[\W_]+$', '', s)
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s)
    # unicode normalize
    s = unicodedata.normalize("NFKD", s)
    return s

def _canonical_key(s):
    """Lowercase canonical key for grouping similar nodes"""
    return re.sub(r'\W+', '', _normalize_text(s)).lower()

def _standardize_relation(rel):
    if rel is None:
        return ""
    rl = str(rel).strip().lower()
    if rl in RELATION_SYNONYMS:
        return RELATION_SYNONYMS[rl]
    return rl

def _detect_columns(df):
    """Return (subj_col, rel_col, obj_col) using case-insensitive detection or None."""
    if df is None:
        return (None, None, None)
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    subj = low.get("subject", None) or low.get("s", None) or (cols[0] if cols else None)
    rel = low.get("relation", None) or low.get("verb", None) or (cols[1] if len(cols) > 1 else None)
    obj = low.get("object", None) or low.get("o", None) or (cols[2] if len(cols) > 2 else None)
    return (subj, rel, obj)

def clean_graph(df_rel):
    """
    Clean the relations DataFrame and return a cleaned copy.
    Input: DataFrame with at least (subject, relation, object)-like columns (auto-detected).
    Output: cleaned DataFrame with columns renamed to: subject, relation, object
    """
    if df_rel is None or df_rel.empty:
        return df_rel

    subj_col, rel_col, obj_col = _detect_columns(df_rel)
    if not subj_col or not rel_col or not obj_col:
        raise ValueError("Could not detect subject/relation/object columns in DataFrame")

    # copy and normalize basic fields
    df = df_rel.copy()
    df['__sub_raw'] = df[subj_col].astype(str).fillna("").apply(lambda x: x.strip())
    df['__obj_raw'] = df[obj_col].astype(str).fillna("").apply(lambda x: x.strip())
    df['__rel_raw'] = df[rel_col].astype(str).fillna("").apply(lambda x: x.strip())

    # create canonical keys for merging
    df['__sub_key'] = df['__sub_raw'].apply(_canonical_key)
    df['__obj_key'] = df['__obj_raw'].apply(_canonical_key)

    # remove orphan or empty subject/object
    df = df[(df['__sub_key'] != "") & (df['__obj_key'] != "")]

    # remove self-loops (subject == object)
    df = df[df['__sub_key'] != df['__obj_key']]

    # standardize relation
    df['__rel_std'] = df['__rel_raw'].apply(_standardize_relation)

    # remove low-value generic triples where either subject or object is generic word
    df = df[~df['__sub_key'].isin([_canonical_key(g) for g in GENERIC_ENTITIES])]
    df = df[~df['__obj_key'].isin([_canonical_key(g) for g in GENERIC_ENTITIES])]

    # collapse duplicates (by keys + relation)
    # keep the first original casing for subject/object/relation
    df = df.drop_duplicates(subset=['__sub_key', '__rel_std', '__obj_key'])

    # Map canonical key back to a preserved display string: pick the most frequent original form
    def pick_display(mapping_series):
        # mapping_series: series of raw strings
        counts = mapping_series.value_counts()
        if len(counts) == 0:
            return ""
        return counts.index[0]

    sub_display = df.groupby('__sub_key')['__sub_raw'].agg(pick_display).to_dict()
    obj_display = df.groupby('__obj_key')['__obj_raw'].agg(pick_display).to_dict()

    # Build cleaned DataFrame
    cleaned = pd.DataFrame({
        'subject': [sub_display.get(k, k) for k in df['__sub_key']],
        'relation': df['__rel_std'].astype(str).tolist(),
        'object': [obj_display.get(k, k) for k in df['__obj_key']],
    })

    # final dedupe and index reset
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # optional: remove rows where relation is empty
    cleaned = cleaned[cleaned['relation'].astype(str).str.strip() != ""]

    return cleaned
