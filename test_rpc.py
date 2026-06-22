import asyncio
import os
from dotenv import load_dotenv
from supabase.client import Client, create_client
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-large-en-v1.5",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

def test_query():
    try:
        vec = embeddings.embed_query("alumni 2025")
        res = supabase_client.rpc(
            "match_documents",
            {
                "query_embedding": vec,
                "match_count": 5,
                "filter": {"source": "alumni"}
            }
        ).execute()
        print("SUCCESS:", len(res.data), "documents returned.")
    except Exception as e:
        print("ERROR:", str(e))

test_query()
