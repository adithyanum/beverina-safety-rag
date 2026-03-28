import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import re

embedder = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("./rag/index.faiss")

with open("./rag/metadata.json") as f:
    metadata = json.load(f)

with open("./data/products.json", "r") as f:
    products_db = json.load(f)["products"]

with open("./data/ingredients.json", "r") as f:
    additives_list = json.load(f)["additives"]
    additives_db = {a['id']: a for a in additives_list}

def retrieve(query, k=30, threshold=1.05):
    
    query = re.sub(r"[^A-Za-z ]", " ", query)
    q_emb = embedder.encode([query])
    distances, indices = index.search(q_emb, k)
    
    # FIX: Create a new list that ONLY keeps items meeting the threshold
    raw_results = []
    for i, dist in zip(indices[0], distances[0]):
        if dist < threshold: # Only keep good matches
            print(f"DEBUG: Found {metadata[i]['name']} with distance {dist:.4f}")
            raw_results.append(metadata[i])
            
    # If nothing met the threshold, return empty immediately
    if not raw_results:
        return []

    # ... The rest of your IDENTIFY INTENT logic ...

    # 1. IDENTIFY INTENT
    target_product = next((r for r in raw_results if r['type'] == 'product'), None)
    
    final_context = []
    
    if target_product:
        p_id = target_product['id']
        p_data = products_db.get(p_id, {})
        # Get the actual ingredient list for this product
        p_ingredients = [ing.lower() for ing in p_data.get('ingredients', [])]
        
        # Add the Product itself to context
        target_product['full_data'] = p_data
        final_context.append(target_product)
        
        # 2. FILTER: Only add additives that are in the list
        for res in raw_results:
            if res['type'] == 'additive':
                a_id = res['id'].lower()
                a_name = res['name'].lower()
                
                # Check if the additive ID or name appears anywhere in the ingredients list
                #
                is_present = any(a_id in ing or a_name in ing for ing in p_ingredients)
                
                if is_present:
                    res['full_data'] = additives_db.get(res['id'])
                    final_context.append(res)
    else:
        # Fallback for general additive queries
        final_context = raw_results

    return final_context[:5]
# -------------------------
# Test
# -------------------------
query = "is pepsi better than coca cola?"
results = retrieve(query)

for r in results:
    print(json.dumps(r, indent=2))
