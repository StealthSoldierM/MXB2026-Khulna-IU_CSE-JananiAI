from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
from google import genai
from dotenv import load_dotenv
import os

MODEL = "models/text-embedding-004"

load_dotenv()

app = FastAPI()


app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))


class ChatResponse(BaseModel):
    response: str
    sources: list

class ChatRequest(BaseModel):
    message: str
    match_count: int = 3

def embedding_task(txt: str):
    try:
        resp = client.models.embed_content(model=MODEL, contents=txt, config=genai.types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"))
        return resp.embeddings[0].values
    except Exception as e:
        return f"got error: {e}"

def search_knowledge(q: str, match_cnt: int):
    q_embed = embedding_task(q)

    res = supabase.rpc('match_knowledge', {
        "query_embedding": q_embed,
        "match_count": match_cnt
    }).execute()

    return res.data

def generate_response(ctx: str, q: str):
    system_prompt = f'''
        You are a compassionate maternal health assistant.
        Do not diagnose.
        Use the following trusted information:
        {ctx}

        Question: {q}
        '''

    resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=system_prompt,
        config=genai.types.GenerateContentConfig(temperature=0.7, max_output_tokens=1024,)
    )
    return resp


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Chat API is running"}

@app.get("/health")
def check():
    return {"status": "healthy"}

@app.post('/chat', response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        # the context
        results = search_knowledge(req.message, req.match_count)
        if not results:
            raise HTTPException(status_code=404, detail="No info found")

        context = '\n'.join([r["content"] for r in results])

        ans_resp = generate_response(context, req.message)

        source = [ {"content": ctx["content"], "similarity": ctx.get("similarity", 0)} for ctx in results]
        return ChatResponse(response=ans_resp, sources=source)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
