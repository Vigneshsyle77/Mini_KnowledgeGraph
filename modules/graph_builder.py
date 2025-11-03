from pyvis.network import Network
import networkx as nx
import pandas as pd

# Load dataset
triples = pd.read_csv("data/extracted_triples.csv")

# Create directed graph
G = nx.DiGraph()

# Fixed node colors
subj_color = "lightblue"
obj_color = "lightgreen"

# Add nodes and edges with color and weights
for _, row in triples.iterrows():
    subj = row["Subject"]
    obj = row["Object"]
    rel = row["Relation"]

    # Add subject/object nodes (bigger if more connections)
    G.add_node(subj, color=subj_color, title=f"Subject: {subj}")
    G.add_node(obj, color=obj_color, title=f"Object: {obj}")
    G.add_edge(subj, obj, label=rel, value=len(rel) / 3)

# Compute node sizes by degree
for node in G.nodes():
    degree = G.degree(node)
    G.nodes[node]["size"] = 10 + degree * 3  # base size + connections

print(f"✅ Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges.\n")

# Print relations for all nodes
print("📘 All Relations in Graph:\n")
for node in G.nodes():
    if len(G[node]) > 0:
        print(f"🔹 {node}:")
        for neighbor in G[node]:
            print(f"   → {neighbor} | Relation: {G[node][neighbor]['label']}")
        print()

# 🌐 Create PyVis Network
net = Network(
    height="800px",
    width="100%",
    directed=True,
    bgcolor="#1e1e1e",
    font_color="white"
)

# ✨ Physics for large graphs — better spacing
net.repulsion(
    node_distance=250,   # how far apart nodes stay
    central_gravity=0.25,
    spring_length=200,
    spring_strength=0.05,
    damping=0.9
)

# Add nodes & edges to PyVis
for node, data in G.nodes(data=True):
    net.add_node(node, label=node, color=data["color"], title=data["title"], size=data["size"])

for u, v, data in G.edges(data=True):
    net.add_edge(u, v, label=data["label"], title=data["label"], value=data["value"])

# Save interactive graph
net.save_graph("ui/expanded_knowledge_graph.html")
print("🌐 Graph saved as 'expanded_knowledge_graph.html' — open it in your browser for the full network view!")
