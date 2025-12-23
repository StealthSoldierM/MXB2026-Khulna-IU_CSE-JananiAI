from supabase import create_client
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

def embedding_task(txt):
    res = genai.embed_content("models/embedding-001", content=txt, task_type="retrieval_document")
    return res["embedding"]

with open("./knowledges-01.txt") as f:
    for line in f:
        if line.strip():
            vec = embedding_task(line.strip())
            supabase.table("knowledge").insert({
                "content": line.strip(),
                "embedding": vec,
            }).execute()

print("knowledge uploaded")
