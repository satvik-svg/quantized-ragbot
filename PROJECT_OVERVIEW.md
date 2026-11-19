# Project Overview: Offline NCERT RAG Chatbot

## What This Project Is

**Offline NCERT RAG Chatbot** (`OfflineChatbot`) is a privacy-focused, fully offline mobile question-answering application built with React Native for **NCERT Maths and Science (Classes 6–8)**.

Instead of relying on cloud APIs, the app runs a complete **Retrieval-Augmented Generation (RAG)** pipeline directly on the user's device:

1. **Offline Data Preparation (Python):** Textbook PDFs are extracted, cleaned, chunked, and embedded into vector representations, then exported as a JSON knowledge base (`app_data.json`).
2. **On-Device Retrieval (ONNX Runtime + Cosine Similarity):** User questions are tokenized and embedded locally using an ONNX-exported `all-MiniLM-L6-v2` model, then matched against the precomputed chunk embeddings via cosine similarity.
3. **On-Device Answer Generation (`llama.rn` / `llama.cpp`):** The top-3 most relevant textbook chunks are injected into a ChatML prompt and passed to a quantized local LLM (`qwen2.5-0.5b-instruct-q4_k_m.gguf`) to generate an answer completely offline.

---

## Repository Structure

```text
.
├── app/                                  # React Native mobile app (Android & iOS scaffolding)
│   ├── App.tsx                           # Chat UI, state management, and service initialization
│   ├── package.json                      # Dependencies (llama.rn, onnxruntime-react-native, react-native-fs)
│   ├── app.json                          # App metadata ("OfflineChatbot")
│   ├── src/
│   │   └── services/
│   │       └── ChatService.ts            # On-device RAG pipeline (asset copy, tokenization, embeddings, LLM)
│   └── android/app/src/main/assets/      # Bundled runtime assets (vocab.txt, ggml-hexagon libs, indices/, models/)
├── preprocessing/                        # Offline Python pipeline for PDFs & model export
│   ├── requirements.txt                  # Python dependencies (pymupdf, sentence-transformers, faiss-cpu, numpy, tqdm)
│   ├── preprocess_and_index.py           # Extracts PDF text, chunks it, builds FAISS index & metadata
│   ├── export_for_app.py                 # Reconstructs vectors from FAISS and exports app_data.json
│   ├── export_onnx.py                    # Exports sentence-transformers/all-MiniLM-L6-v2 to ONNX via optimum
│   └── export_onnx_instructions.py       # CLI instructions for optimum-based ONNX export
├── books/                                # Input directory for NCERT PDF textbooks
├── assets/
│   ├── indices/                          # Intermediate FAISS index & exported app_data.json output
│   └── models/                           # Exported ONNX embedding model output directory
└── README.md                             # Setup and build instructions
```

---

## Architecture & Data Flow

### 1. Offline Preprocessing Pipeline (`preprocessing/`)
- **`preprocess_and_index.py`**:
  - Reads `.pdf` files from `books/` using **PyMuPDF** (`fitz`).
  - Cleans text and splits it into overlapping word chunks (`chunk_size = 300`, `overlap = 50`, skipping chunks `<= 50` words).
  - Generates 384-dimensional embeddings using `SentenceTransformer('all-MiniLM-L6-v2')`.
  - Stores embeddings in a **FAISS** `IndexFlatL2` index (`assets/indices/ncert.index`), raw chunks in `chunks.jsonl`, and chunk text/source metadata in `chunks_meta.pkl`.
- **`export_for_app.py`**:
  - Loads `ncert.index` and `chunks_meta.pkl`, reconstructs the embedding vectors from FAISS, and combines them with chunk text and source names into `assets/indices/app_data.json`.
- **`export_onnx.py` / `export_onnx_instructions.py`**:
  - Exports `sentence-transformers/all-MiniLM-L6-v2` and its tokenizer to ONNX (`model.onnx` + `vocab.txt`) using Hugging Face `optimum` (`ORTModelForFeatureExtraction`).

### 2. Mobile Application (`app/`)
- **`app/App.tsx`**:
  - Initializes `ChatService.init()` on startup (showing a `"Loading Brain..."` indicator while models load).
  - Provides a chat interface (`FlatList` message history, `TextInput`, and send button) that sends user queries to `ChatService.chat()`.
- **`app/src/services/ChatService.ts`**:
  - **Initialization (`init`)**: Copies `qwen2.5-0.5b-instruct-q4_k_m.gguf`, `model.onnx`, and `vocab.txt` from bundled native assets to `RNFS.DocumentDirectoryPath` if not already present, parses `vocab.txt`, initializes `llama.rn` (`n_ctx: 2048`, `n_threads: 4`), and creates an `onnxruntime-react-native` `InferenceSession`.
  - **Query Embedding (`getEmbedding`)**: Tokenizes the query using a simplified WordPiece/vocab lookup (`[CLS]`, `[UNK]`, `[SEP]`), runs the ONNX session with `input_ids`, `attention_mask`, and `token_type_ids`, and applies mean pooling across the 384-dimensional hidden states.
  - **Semantic Retrieval & Generation (`chat`)**: Computes cosine similarity between the query embedding and every entry in `app_data.json`, selects the top 3 chunks, formats a `<|im_start|>system ... <|im_start|>user ... <|im_start|>assistant` prompt, and generates up to 400 tokens via `llamaContext.completion()`.

---

## Core Technologies

| Layer | Technology / Library | Role in Project |
| :--- | :--- | :--- |
| **Mobile Framework** | React Native `0.82.1`, React `19.1.1`, TypeScript | Cross-platform mobile UI (`App.tsx`) and app logic |
| **On-Device LLM** | `llama.rn` (`^0.9.0`) + `Qwen2.5-0.5B-Instruct` (Q4_K_M GGUF) | Local text generation (`n_ctx: 2048`, `n_threads: 4`) |
| **On-Device Embeddings** | `onnxruntime-react-native` (`^1.23.2`) + `all-MiniLM-L6-v2` (ONNX) | Local query vector generation (384-dim) |
| **Storage / Assets** | `react-native-fs` (`^2.20.0`), bundled JSON (`app_data.json`) | Copying model assets to device storage and reading vocab |
| **Preprocessing** | Python (`pymupdf`, `sentence-transformers`, `faiss-cpu`, `numpy`, `tqdm`) | PDF extraction, chunking, embedding, and FAISS indexing |
| **Model Conversion** | Hugging Face `optimum` & `transformers` | Exporting `all-MiniLM-L6-v2` to ONNX format |
