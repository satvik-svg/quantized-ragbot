# Offline NCERT RAG Chatbot

A privacy-focused mobile chatbot for **NCERT Maths and Science, Classes 6–8**.

The application runs completely offline using an on-device LLM and Retrieval-Augmented Generation (RAG). Questions are matched against indexed NCERT textbook content, and answers are generated locally without sending user data to external servers.

## Features

* Fully offline inference
* On-device RAG pipeline
* NCERT Maths and Science support
* Semantic search using vector embeddings
* Quantized LLM for mobile devices
* Local document storage
* Android-optimized React Native application

## Tech Stack

* **Mobile App:** React Native, TypeScript
* **LLM Inference:** `llama.rn` with `llama.cpp`
* **LLM:** Qwen2.5-0.5B-Instruct GGUF
* **Embeddings:** all-MiniLM-L6-v2 ONNX
* **Vector Runtime:** `onnxruntime-react-native`
* **Tokenizer:** Custom WordPiece tokenizer in TypeScript
* **Preprocessing:** Python, LangChain, FAISS
* **Android Build:** Gradle, NDK, CMake

## How It Works

```text
User Question
      ↓
Generate Query Embedding
      ↓
Search Indexed NCERT Content
      ↓
Retrieve Relevant Text Chunks
      ↓
Send Context to Local LLM
      ↓
Generate Offline Answer
```

## Project Structure

```text
offline-cb/
├── app/
│   ├── android/
│   │   └── app/src/main/assets/
│   ├── src/
│   │   ├── services/
│   │   │   └── ChatService.ts
│   │   └── App.tsx
│   └── package.json
├── preprocessing/
│   ├── preprocess_and_index.py
│   └── requirements.txt
├── books/
└── README.md
```

## Prerequisites

* Node.js 18+
* Python 3.10+
* Java JDK 17
* Android Studio
* Android SDK, NDK, and CMake
* Android device with USB debugging enabled

## Setup

### 1. Preprocess NCERT Books

Place the NCERT PDF files inside:

```text
books/
```

Install preprocessing dependencies:

```bash
cd preprocessing
pip install -r requirements.txt
```

Generate text chunks and embeddings:

```bash
python preprocess_and_index.py
```

This generates:

```text
app_data.json
```

### 2. Download Models

Required files:

```text
qwen2.5-0.5b-instruct-q4_k_m.gguf
model.onnx
vocab.txt
```

Recommended models:

* **LLM:** Qwen2.5-0.5B-Instruct Q4_K_M GGUF
* **Embedding Model:** all-MiniLM-L6-v2 ONNX

### 3. Add Android Assets

Place all models and indexed data inside:

```text
app/android/app/src/main/assets/
```

Expected structure:

```text
assets/
├── qwen2.5-0.5b-instruct-q4_k_m.gguf
├── model.onnx
├── vocab.txt
└── indices/
    └── app_data.json
```

### 4. Run the Application

```bash
cd app
npm install
npx react-native run-android
```

## Usage

1. Launch the application.
2. Wait for the local models to initialize.
3. Ask a Maths or Science question.
4. The app retrieves relevant NCERT content and generates an answer locally.

Example:

```text
What is photosynthesis?
```

## Troubleshooting

### Clean Android Build

```bash
cd app/android
./gradlew clean
```

On Windows:

```bash
gradlew.bat clean
```

### Slow Inference

Performance depends on the device CPU and RAM. The 0.5B quantized model is designed for mid-range Android devices.

### Storage Issues

The bundled models can require several hundred megabytes. Keep at least 2 GB of free storage available during installation and build.

## License

Licensed under the MIT License.
