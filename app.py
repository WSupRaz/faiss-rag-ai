import requests
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

model = SentenceTransformer("all-MiniLM-L6-v2")

reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

def chunk_text(text, size=300):
    return [text[i:i+size] for i in range(0, len(text), size)]

chunks = chunk_text(text)

embeddings = model.encode(chunks)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

question = input("Ask question: ")

q_embedding = model.encode([question])

k = 3

distances, indices = index.search(np.array(q_embedding), k)

top_chunks = [chunks[i] for i in indices[0]]

context = "\n".join(top_chunks)

data = {
    "model": "openai/gpt-3.5-turbo",
    "messages": [
        {"role": "system", "content": "Answer only using the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
}

response = requests.post(url, headers=headers, json=data)

result = response.json()

if "choices" in result:
    answer = result["choices"][0]["message"]["content"]
    print("\nAI Answer:\n", answer)
else:
    print("API Error:", result)