import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------
# Load embedding model
# -------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
metadata = []

# -------------------------
# Load PRODUCTS
# -------------------------
with open("./data/products.json", "r") as f:
    products = json.load(f)["products"]

for pid, product in products.items():
    text = (
        f"Product name: {product['name']}. "
        f"Category: {product['category']}. "
        f"Ingredients: {', '.join(product['ingredients'])}. "
        f"Nutrition per 100ml: "
        f"Energy {product['nutrition_per_100ml']['energy_kcal']} kcal, "
        f"Sugar {product['nutrition_per_100ml']['sugar_g']} g, "
        f"Sodium {product['nutrition_per_100ml']['sodium_mg']} mg, "
        f"Caffeine {product['nutrition_per_100ml']['caffeine_mg']} mg."
    )
    documents.append(text)
    metadata.append({
        "type": "product",
        "id": pid,
        "name": product["name"]
    })

# -------------------------
# Load ADDITIVES
# -------------------------
with open("./data/ingredients.json", "r") as f:
    additives = json.load(f)["additives"]

for additive in additives:
    text = (
        f"Additive name: {additive['name']}. "
        f"Category: {additive['category']}. "
        f"Health effects: {' '.join(additive['health_effects'])}. "
        f"Risk level: {additive['risk_level']}. "
        f"Child safety: {additive.get('child_safety_level', 'Not specified')}. "
        f"Avoid if: {', '.join(additive.get('avoid_if', []))}."
    )
    documents.append(text)
    metadata.append({
        "type": "additive",
        "id": additive["id"],
        "name": additive["name"]
    })

# -------------------------
# Create embeddings
# -------------------------
print("Embedding documents...")
embeddings = embedder.encode(documents, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# -------------------------
# Save index + metadata
# -------------------------
faiss.write_index(index, "rag/index.faiss")

with open("rag/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"RAG index built successfully with {len(documents)} entries.")
