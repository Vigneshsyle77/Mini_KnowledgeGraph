import pandas as pd
import spacy
 
nlp = spacy.load("en_core_web_sm")
 
# Load dataset
data = pd.read_csv("data/cross_domain_article.csv")
 
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
 
# Apply NER to each sentence
data["entities"] = data["text"].apply(lambda x: extract_entities(str(x)) if pd.notnull(x) else [])
 
# Display sample
print(data.head())
 
# Save extracted entities
data.to_csv("data/processed_entities.csv", index=False)
print("✅ NER extraction complete — saved to data/processed_entities.csv")