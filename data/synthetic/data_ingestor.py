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
        resp = client.models.embed_content(model=MODEL, contents=txt, config=genai.types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768))
        return [embedding.values for embedding in resp.embeddings]
    except Exception as e:
        print(f"Error embedding batch {e}")
        raise

def process_in_batch():
    with open("./knowledges-01.txt") as f:
        lines = [l.strip() for l in f if l.strip()]
    
    line_count = len(lines)

    print(f"total: {line_count} is to process")

    for i in range(0, line_count, BATCH_SIZE):
        batch = lines[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batch = (line_count + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"Processing batch {batch_num}/{total_batch} ({len(batch)}) items...")

        try:
            embed = embedding_task(batch)

            rec = [
                    {"content": txt, "embedding": emb} for txt, emb in zip(batch, embed)
                ]
            supabase.table("knowledge").insert(rec).execute()
            print(f"batch {batch_num} uploaded successfully")

            if i + BATCH_SIZE < line_count:
                print(f"waiting {RATE_DELAY} for rate limit...")
                time.sleep(RATE_DELAY)
        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
            print("waiting 1 min")
            time.sleep(60)
            continue

if __name__ == "__main__":
    process_in_batch()
    print("knowledge uploaded")
