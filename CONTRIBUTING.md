# Contributing to Offline NCERT RAG Chatbot

Thank you for your interest in contributing! This guide covers local setup, project conventions, and how to submit changes.

## Repository Structure

- `app/` — React Native (`0.82.1`) TypeScript mobile app (`OfflineChatbot`)
  - `app/App.tsx` — Chat UI
  - `app/src/services/ChatService.ts` — On-device RAG pipeline (`llama.rn` + `onnxruntime-react-native`)
  - `app/android/` — Android Gradle project (`minSdkVersion 24`, `compileSdkVersion 36`, `ndkVersion 27.1.12297006`)
  - `app/ios/` — iOS Xcode project & `Podfile`
  - `app/__tests__/` — Jest unit tests
- `preprocessing/` — Python scripts for PDF chunking, FAISS indexing, and ONNX export
- `books/` — Input NCERT PDF directory (PDFs are git-ignored)
- `assets/` — Intermediate preprocessing outputs (`models/`, `indices/`; large files are git-ignored)

## Prerequisites

- **Node.js** `>= 20` (per `app/package.json`) and **npm**
- **Python** `3.10+` (for `preprocessing/`)
- **Android**: Java JDK 17, Android Studio, Android SDK (API 36), NDK (`27.1.12297006`), and CMake
- **iOS** (macOS only): Xcode, Ruby `>= 2.6.10`, and Bundler/CocoaPods (see `app/Gemfile`)

## Development Setup

### 1. Preprocessing (Optional — needed to rebuild indices/embeddings)

Place source PDFs in `books/`, then run from `preprocessing/`:

```bash
cd preprocessing
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Extract text, chunk, and build FAISS index in ../assets/indices/
python preprocess_and_index.py

# Export chunks + embeddings to ../assets/indices/app_data.json
python export_for_app.py
```

To export the `sentence-transformers/all-MiniLM-L6-v2` embedding model to ONNX:

```bash
pip install "optimum[exporters]"
python export_onnx.py
```

### 2. Model & Asset Placement

`app/src/services/ChatService.ts` expects:
- Bundled native assets (`app/android/app/src/main/assets/` for Android, or Xcode bundle resources for iOS):
  - `qwen2.5-0.5b-instruct-q4_k_m.gguf`
  - `model.onnx`
  - `vocab.txt` (already present in `app/android/app/src/main/assets/vocab.txt`)
- Imported JSON index for TypeScript bundling:
  - `app/src/assets/indices/app_data.json`

### 3. Mobile App Setup & Run

```bash
cd app
npm install
```

Start Metro and run on a device/emulator:

```bash
npm start
# In another terminal:
npm run android
# Or on macOS for iOS:
bundle install && (cd ios && bundle exec pod install)
npm run ios
```

## Code Quality & Checks

Run all checks from `app/` before opening a pull request:

```bash
cd app
npm run lint          # ESLint (@react-native config)
npx tsc --noEmit      # TypeScript type-check
npm test              # Jest test suite
```

### Style Conventions

- **TypeScript / React Native**: Follow `app/.eslintrc.js` (`@react-native`) and `app/.prettierrc.js` (single quotes, trailing commas, `arrowParens: 'avoid'`).
- **Python**: Keep preprocessing scripts compatible with `preprocessing/requirements.txt` and use relative paths consistent with running inside `preprocessing/`.

## Submitting Changes

1. Create a feature branch from `main` (e.g. `feat/wordpiece-tokenizer` or `fix/android-asset-copy`).
2. **Never commit large binaries or raw data** — `.gitignore` excludes `*.pdf`, `*.gguf`, `*.onnx`, `*.bin`, `*.tflite`, `*.index`, `*.pkl`, `*.jsonl`, and `app_data.json`. Preserve `.gitkeep` files in asset directories.
3. Add or update tests in `app/__tests__/` when changing app logic or UI.
4. Keep commits focused with clear messages, and describe what changed and how it was tested in your pull request.
