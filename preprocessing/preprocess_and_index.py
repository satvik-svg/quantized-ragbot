import os
import glob
import json
import re
import fitz  # pymupdf
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration
BOOKS_DIR = "../books"
ASSETS_DIR = "../assets/indices"
CHUNKS_FILE = os.path.join(ASSETS_DIR, "chunks.jsonl")
INDEX_FILE = os.path.join(ASSETS_DIR, "ncert.index")
META_FILE = os.path.join(ASSETS_DIR, "chunks_meta.pkl")

# Model for embeddings (Small, fast, good for mobile)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

def clean_text(text):
    """Basic text cleaning."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable characters
    text = "".join(ch for ch in text if ch.isprintable())
    return text.strip()

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    full_text = ""
    print(f"Extracting text from {os.path.basename(pdf_path)}...")
    for page in tqdm(doc, desc="Pages"):
        full_text += page.get_text() + "\n"
    return full_text

def chunk_text(text, source_name, chunk_size=300, overlap=50):
    """Splits text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_str = " ".join(chunk_words)
        if len(chunk_words) > 50:  # Skip very small chunks
            chunks.append({
                "id": f"{source_name}_{i}",
                "text": chunk_str,
                "source": source_name
            })
    return chunks

def main():
    # Ensure output directory exists
    os.makedirs(ASSETS_DIR, exist_ok=True)

    all_chunks = []

    # 1. Extract and Chunk
    pdf_files = glob.glob(os.path.join(BOOKS_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {BOOKS_DIR}")
        return

    for pdf_file in pdf_files:
        source_name = os.path.basename(pdf_file)
        raw_text = extract_text_from_pdf(pdf_file)
        cleaned_text = clean_text(raw_text)
        file_chunks = chunk_text(cleaned_text, source_name)
        all_chunks.extend(file_chunks)
        print(f"Processed {source_name}: {len(file_chunks)} chunks.")

    print(f"Total chunks generated: {len(all_chunks)}")

    # Save chunks to JSONL (useful for debugging or re-indexing)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")
    
    # 2. Generate Embeddings
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    texts = [c["text"] for c in all_chunks]
    print("Generating embeddings (this may take a while)...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    # Convert to float32 for FAISS
    embeddings = np.array(embeddings).astype('float32')

    # 3. Build FAISS Index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # Brute force L2 distance (exact search)
    index.add(embeddings)
    
    # Save Index
    faiss.write_index(index, INDEX_FILE)
    print(f"Index saved to {INDEX_FILE}")

    # Save Metadata (to map index ID back to text)
    # We only need the text and source for the app
    metadata = [{"text": c["text"], "source": c["source"]} for c in all_chunks]
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to {META_FILE}")

    print("\n--- Preprocessing Complete ---")
    print(f"1. Index File: {INDEX_FILE} (Bundle this in App)")
    print(f"2. Metadata File: {META_FILE} (Bundle this in App)")
    print(f"3. Embedding Model: You will need to export '{EMBEDDING_MODEL_NAME}' to ONNX for the app.")

if __name__ == "__main__":
    main()
