import pickle
import json
import numpy as np
import faiss

# Paths
ASSETS_DIR = "../assets/indices"
INDEX_FILE = f"{ASSETS_DIR}/ncert.index"
META_FILE = f"{ASSETS_DIR}/chunks_meta.pkl"
APP_DATA_FILE = "../assets/indices/app_data.json"

print("Loading index and metadata...")
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    metadata = pickle.load(f)

# Extract embeddings from FAISS index
# For IndexFlatL2, we can access the raw vectors
try:
    embeddings = index.reconstruct_n(0, index.ntotal)
except:
    # Fallback if reconstruct is not supported (unlikely for FlatL2)
    print("Could not reconstruct embeddings from index directly.")
    exit(1)

print(f"Exporting {len(metadata)} chunks to {APP_DATA_FILE}...")

export_data = []
for i, meta in enumerate(metadata):
    export_data.append({
        "id": i,
        "text": meta["text"],
        "source": meta["source"],
        "embedding": embeddings[i].tolist() # Convert numpy array to list
    })

with open(APP_DATA_FILE, "w", encoding="utf-8") as f:
    json.dump(export_data, f)

print("Done! app_data.json contains text + embeddings.")
print("Size might be large (text + float vectors).")
# 1500 chunks * (300 words + 384 floats) -> ~5-10MB. Acceptable.
