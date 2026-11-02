import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { LlamaContext, initLlama } from 'llama.rn';
import RNFS from 'react-native-fs';
import appData from '../assets/indices/app_data.json';

const MODEL_FILENAME = 'qwen2.5-0.5b-instruct-q4_k_m.gguf';
const EMBEDDING_MODEL_FILENAME = 'model.onnx';
const VOCAB_FILENAME = 'vocab.txt';

class ChatService {
  private llamaContext: LlamaContext | null = null;
  private embeddingSession: InferenceSession | null = null;
  private vocab: Record<string, number> = {};
  private isInitialized = false;

  async init() {
    if (this.isInitialized) return;

    console.log('Initializing ChatService...');

    const modelPath = `${RNFS.DocumentDirectoryPath}/${MODEL_FILENAME}`;
    const embeddingPath = `${RNFS.DocumentDirectoryPath}/${EMBEDDING_MODEL_FILENAME}`;
    const vocabPath = `${RNFS.DocumentDirectoryPath}/${VOCAB_FILENAME}`;

    await this.copyAssetIfNotExists(MODEL_FILENAME, modelPath);
    await this.copyAssetIfNotExists(EMBEDDING_MODEL_FILENAME, embeddingPath);
    await this.copyAssetIfNotExists(VOCAB_FILENAME, vocabPath);

    console.log('Loading Vocab...');
    await this.loadVocab(vocabPath);

    console.log('Loading LLM...');
    this.llamaContext = await initLlama({
      model: modelPath,
      n_ctx: 2048,
      n_threads: 4,
    });

    console.log('Loading Embedding Model...');
    this.embeddingSession = await InferenceSession.create(embeddingPath);

    this.isInitialized = true;
    console.log('ChatService Initialized!');
  }

  private async copyAssetIfNotExists(filename: string, destPath: string) {
    const exists = await RNFS.exists(destPath);
    if (!exists) {
      console.log(`Copying ${filename} to ${destPath}...`);
      if (RNFS.copyFileAssets) {
        await RNFS.copyFileAssets(filename, destPath);
      } else {
        await RNFS.copyFile(RNFS.MainBundlePath + '/' + filename, destPath);
      }
    }
  }

  private async loadVocab(path: string) {
    const content = await RNFS.readFile(path, 'utf8');
    const lines = content.split('\n');
    lines.forEach((line, index) => {
      if (line.trim()) {
        this.vocab[line.trim()] = index;
      }
    });
  }

  private tokenize(text: string): number[] {
    const tokens: number[] = [];
    tokens.push(this.vocab['[CLS]'] || 101);

    const words = text.toLowerCase().split(/\s+/);
    for (const word of words) {
      let current = word;
      // Basic WordPiece tokenization (simplified)
      if (this.vocab[current]) {
        tokens.push(this.vocab[current]);
      } else {
        // Unknown word handling - simplified: just skip or use UNK
        tokens.push(this.vocab['[UNK]'] || 100);
      }
    }

    tokens.push(this.vocab['[SEP]'] || 102);
    return tokens;
  }

  private async getEmbedding(text: string): Promise<number[]> {
    if (!this.embeddingSession) throw new Error('Embedding session not initialized');

    const inputIds = this.tokenize(text);
    const attentionMask = new Array(inputIds.length).fill(1);
    const tokenTypeIds = new Array(inputIds.length).fill(0);

    // Pad to fixed length if needed, or just use dynamic length (ONNX usually supports dynamic)
    // For batching, padding is needed. For single inference, usually not strictly required if model supports it.
    // But let's pad to be safe or just pass as is.
    
    const inputTensor = new Tensor('int64', new BigInt64Array(inputIds.map(BigInt)), [1, inputIds.length]);
    const maskTensor = new Tensor('int64', new BigInt64Array(attentionMask.map(BigInt)), [1, attentionMask.length]);
    const typeTensor = new Tensor('int64', new BigInt64Array(tokenTypeIds.map(BigInt)), [1, tokenTypeIds.length]);

    const feeds = {
      input_ids: inputTensor,
      attention_mask: maskTensor,
      token_type_ids: typeTensor,
    };

    const results = await this.embeddingSession.run(feeds);
    // Output is usually 'last_hidden_state' or 'pooler_output' or similar.
    // For sentence-transformers, we often want mean pooling of last_hidden_state.
    // Let's check the output names. Usually 'last_hidden_state' is output[0].
    
    const output = results[Object.keys(results)[0]]; // Get first output
    const data = output.data as Float32Array;
    
    // Perform Mean Pooling
    // data is [1, seq_len, hidden_size] (flattened)
    const hiddenSize = 384; // MiniLM-L6-v2
    const seqLen = inputIds.length;
    
    const embedding = new Array(hiddenSize).fill(0);
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < hiddenSize; j++) {
        embedding[j] += data[i * hiddenSize + j];
      }
    }
    
    // Normalize
    for (let j = 0; j < hiddenSize; j++) {
      embedding[j] /= seqLen;
    }

    return embedding;
  }

  private cosineSimilarity(a: number[], b: number[]) {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  async chat(question: string): Promise<string> {
    if (!this.llamaContext) throw new Error('LLM not initialized');

    // 1. Retrieve Context
    let contextText = "";
    try {
        const qEmb = await this.getEmbedding(question);
        const relevantChunks = appData
          .map((chunk: any) => ({ ...chunk, score: this.cosineSimilarity(qEmb, chunk.embedding) }))
          .sort((a: any, b: any) => b.score - a.score)
          .slice(0, 3);
        
        contextText = relevantChunks.map((c: any) => c.text).join('\n\n');
        console.log("Retrieved Context:", contextText);
    } catch (e) {
        console.error("Embedding failed, using fallback context", e);
        contextText = "Could not retrieve context. Please answer based on general knowledge.";
    }

    // 2. Construct Prompt
    const prompt = `<|im_start|>system
You are a helpful assistant for NCERT Maths and Science. Use the following context to answer the user's question.
Context:
${contextText}
<|im_end|>
<|im_start|>user
${question}
<|im_end|>
<|im_start|>assistant
`;

    // 3. Generate
    const response = await this.llamaContext.completion({
      prompt,
      n_predict: 400,
      stop: ['<|im_end|>'],
    });

    return response.text;
  }
}

export default new ChatService();
