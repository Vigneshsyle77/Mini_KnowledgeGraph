import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load your dataset
data = pd.read_csv("data/extracted_triples.csv")

# Try to automatically find the column with text
TEXT_COLUMN = 'Text'


if TEXT_COLUMN is None:
    raise ValueError("❌ No suitable text column found! Please check your CSV headers.")

sentences = data[TEXT_COLUMN].dropna().head(50).tolist()  # use first 50 sentences

# Encode dataset sentences
embeddings = model.encode(sentences, convert_to_tensor=True)

# Query from user
query = input("🔍 Enter your query: ")
query_embedding = model.encode(query, convert_to_tensor=True)

# Compute cosine similarity
cosine_scores = util.cos_sim(query_embedding, embeddings)

# Rank and print top 5 results
top_results = cosine_scores[0].cpu().tolist()
ranked_indices = sorted(range(len(top_results)), key=lambda i: top_results[i], reverse=True)[:5]

print("\n🔎 Top 5 Similar Sentences:")
for idx in ranked_indices:
    print(f"{sentences[idx]}  -->  Score: {top_results[idx]:.3f}")