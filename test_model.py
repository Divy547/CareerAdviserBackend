import torch
from sentence_transformers import SentenceTransformer
import pickle
from train import RoleClassifier  # your class
import os

device = torch.device("cpu")

# Load role map
with open("models/role_map.pkl", "rb") as f:
    role_map = pickle.load(f)

# Load model
input_dim = 384
num_classes = len(role_map)
model = RoleClassifier(input_dim, num_classes)
model.load_state_dict(torch.load("models/role_classifier.pth", map_location=device))
model.to(device)
model.eval()

# Example embedding
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
example_text = ["python, pandas, machine learning"]
embedding = embedder.encode(example_text, convert_to_tensor=True).to(device)

with torch.inference_mode():
    logits = model(embedding)
    probs = torch.softmax(logits, dim=1)

print("Predicted probabilities:", probs)
