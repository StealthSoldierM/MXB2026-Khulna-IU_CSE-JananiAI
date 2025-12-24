from supabase import create_client
# deprecated, change this
from google import genai
from dotenv import load_dotenv
import os
import time

load_dotenv()


BATCH_SIZE = 100
RATE_DELAY = 15
MODEL = "models/text-embedding-004"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

def embedding_task(txt):
    try:
        resp = client.models.embed_content(model=MODEL, contents=[txt], config=genai.types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"))
        return resp.embeddings[0].values
    except Exception as e:
        print(f"Error embedding batch {e}")
        raise

def search_knowledge(q):
    q_embed = embedding_task(q)

    res = supabase.rpc('match_knowledge', {
        "query_embedding": q_embed,
        "match_count": 3
    }).execute()

    return res.data

if __name__ == "__main__":
    print(search_knowledge("a woman aged 15 with low education and no happiness, anc visits 2"))
    print("knowledge found")
