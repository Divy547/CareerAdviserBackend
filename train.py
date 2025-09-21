import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "data", "skills_roles.csv")
df = pd.read_csv(csv_path)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

X = embedder.encode(df["skills"].tolist(), convert_to_tensor=True)
X = X.clone().detach().requires_grad_(True)

y = torch.tensor(df["role"].astype("category").cat.codes.values, dtype=torch.long)

# print("Embedding shape:", X.shape)  # (10, 384)
# print("Labels:", y.tolist())

# ! Training the model with pytorch

num_classes = len(set(y.tolist()))
input_dim = X.shape[1]

class RoleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.fc(x)
    
model = RoleClassifier(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
  
role_map = dict(enumerate(df["role"].astype("category").cat.categories)) # * maps the jobs with numeric values

# ! Saving the model and the roleMap
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/role_classifier.pth")

with open("models/role_map.pkl", "wb") as f:
    pickle.dump(role_map, f)

