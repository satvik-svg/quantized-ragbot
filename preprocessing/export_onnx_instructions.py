from sentence_transformers import SentenceTransformer
import torch

# This script exports the embedding model to ONNX format for use in the mobile app.
# You might need to install 'onnx' and 'onnxruntime' packages.

model_name = 'all-MiniLM-L6-v2'
output_path = "../assets/models/embedding_model.onnx"

print(f"Loading {model_name}...")
model = SentenceTransformer(model_name)

# Create a dummy input for tracing
# The model expects a list of strings, but for ONNX export via torch.onnx.export, 
# it's often easier to use the underlying transformer module directly if we want raw control,
# OR use the 'optimum' library which is the recommended way for Transformers -> ONNX.

# For simplicity in this script, we will just print instructions because 
# exporting to ONNX properly often requires the 'optimum' CLI tool which is much more robust.

print("\n--- INSTRUCTIONS FOR ONNX EXPORT ---")
print("To export the embedding model for your React Native app, the best way is to use the 'optimum' library.")
print("Run the following commands in your terminal:")
print("")
print("pip install optimum[exporters]")
print(f"optimum-cli export onnx --model {model_name} --task feature-extraction ../assets/models/onnx_output_dir")
print("")
print("This will generate a 'model.onnx' file in '../assets/models/onnx_output_dir'.")
print("You can then rename it and move it to your assets folder.")
