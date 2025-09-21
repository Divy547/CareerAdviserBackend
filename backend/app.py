from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from dotenv import load_dotenv
import torch
import pickle

load_dotenv()


# ! Loading the model 
import sys
import os
sys.path.append(os.path.abspath(".."))
from train import RoleClassifier

device = torch.device("cpu")

with open("models/role_map.pkl", "rb") as f:
    role_map = pickle.load(f)

# embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu") # CPU

# Load classifier
input_dim = 384
num_classes = len(role_map)
model = RoleClassifier(input_dim, num_classes)
# model.load_state_dict(torch.load("models/role_classifier.pth"))
model.load_state_dict(torch.load("models/role_classifier.pth", map_location=device)) # CPU
model.to(device) # CPU
model.eval()


all_skills = [
    "python", "pandas", "sql", "statistics", "machine learning", "deep learning",
    "excel", "tableau", "data visualization",
    "html", "css", "javascript", "react", "vue",
    "docker", "kubernetes", "aws", "linux", "terraform", "ansible"
]

# Configuring Gemini
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash")


# Accepting a Post Request..
app = FastAPI()

# Handling cors
origins = [
    "http://localhost:3000",
    "https://career-adviser-lovat.vercel.app",
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    user_input : str

@app.post("/predict")
def predict_role(data: UserInput):
    user_text = data.user_input.lower()
    print(user_text)
    skills =  [skill for skill in all_skills if skill in user_text]
    skills_text = ", ".join(skills)

    top_roles, probab = [], []
    if skills:
        # user_embedding = embedder.encode([skills_text], convert_to_tensor=True)
        user_embedding = embedder.encode([skills_text], convert_to_tensor=True).to(device) # CPU

        with torch.inference_mode():
            logits = model(user_embedding)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probs, k=3, dim=1)

        top_roles = [role_map[i.item()] for i in top_indices[0]]
        probab = [p.item() for p in top_probs[0]]



    # 3. Generate personalized advice via Gemini
    # prompt = f"""
    # A student has these skills: {skills_text}.
    # {"The top 3 predicted career roles are: " + str(top_roles) if top_roles else "No model prediction is available."}
    # Please provide:
    # 1. The most suitable role (even if skills are missing, suggest one based on common career paths).
    # 2. Missing skill gaps (with desc even if skills are missing, suggest some based on common career paths).
    # 3. A 5-step learning roadmap (step, title, desc).
    # 4. 3-5 high quality resources (courses, books, websites) (with desc) (resources).
    # Respond in JSON format.
    # """
    prompt = f"""
A student has these skills: {skills_text}.
{"The top 3 predicted career roles are: " + str(top_roles) if top_roles else ""}

Please provide a JSON response with the following predefined fields:

{{
    "most_suitable_role": "string (suggest a role even if skills are missing)",
    "missing_skill_gaps": [
        {{
            "skill": "string (name of missing skill)",
            "description": "string (short description of the skill)"
        }}
    ],
    "learning_roadmap": [
        {{
            "step": 1,
            "title": "string (title of step)",
            "description": "string (description of step)"
        }}
    ],
    "resources": [
        {{
            "name": "string (name of the resource)",
            "type": "string (book/course/website/etc.)",
            "description": "string (short description of resource)",
            "link": "string (URL if available, otherwise leave empty)"
        }}
    ]
}}

Even if no skills are provided, fill all fields with reasonable suggestions.
Respond **only in JSON format**.
"""

    response = llm.generate_content(prompt)


    return {
        "roles": top_roles if top_roles else ["Not predicted"],
        "probabilities": probab if probab else [],
        "gemini_advice": response.text
    }
    
    


    