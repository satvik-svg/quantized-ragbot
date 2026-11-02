from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from pathlib import Path

model_id = "sentence-transformers/all-MiniLM-L6-v2"
output_dir = Path("../assets/models/onnx_output_dir")

print(f"Exporting {model_id} to ONNX...")

# Load model and tokenizer, and export to ONNX
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save the model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model exported to {output_dir}")
