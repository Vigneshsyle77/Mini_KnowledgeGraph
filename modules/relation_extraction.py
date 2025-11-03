import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load triples
triples = pd.read_csv("data/extracted_triples.csv")

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges
for _, row in triples.iterrows():
    subj = row["Subject"]
    obj = row["Object"]
    rel = row["Relation"]
    G.add_node(subj)
    G.add_node(obj)
    G.add_edge(subj, obj, label=rel)

print(f"✅ Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges.")

# Example: show all relations for a sample node
node = "University"
if node in G:
    print(f"\nRelations for '{node}':")
    for neighbor in G[node]:
        print("→", neighbor, "| Relation:", G[node][neighbor]['label'])
else:
    print(f"⚠️ Node '{node}' not found in graph.")
