from fastapi import FastAPI, Depends
from pydantic import BaseModel
from supabase import create_client
from google import genai
from dotenv import load_dotenv
import os

MODEL = "models/text-embedding-004"

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

app = FastAPI()

class Query(BaseModel):
    question: str

def embedding_task(txt: Query):
    try:
        resp = client.models.embed_content(model=MODEL, contents=txt.question, config=genai.types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"))
        return resp.embeddings
    except Exception as e:
        return f"got error: {e}"

def search_knowledge(q: Query):
    q_embed = embedding_task(q)

    res = supabase.rpc('match_knowledge', {
        "query_embedding": q_embed,
        "match_count": 3
    }).execute()

    return res.data

def generate_response(ctx: str, q: Query):
    resp = client.models.generate_content(
        model="gemini-pro",
        contents=f'''
        You are a compassionate maternal health assistant.
        Do not diagnose.
        Use the following trusted information:
        {ctx}

        Question: {q.question}
        '''
    )
    return resp


@app.post('/chat')
def chat(q: Query):
    results = search_knowledge(q)
    context = '\n'.join([r["content"] for r in results])
    resp = generate_response(context, q)
    return {"reply": resp.text}
