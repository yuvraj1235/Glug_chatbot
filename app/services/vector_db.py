import os
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
# 1. FIXED: Changed import to HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Replace the local disk path with your Supabase Cloud credentials
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") # Use the service_role key to bypass RLS for inserts
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# Initialize the official Supabase client
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# 2. FIXED: Instantiate using HuggingFaceEndpointEmbeddings (384 Dimensions)
# Note: Ensure your Supabase table schema is configured for vector(384)!
embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-large-en-v1.5",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

def store_documents_in_supabase(documents):
    # Swap Chroma for SupabaseVectorStore
    vector_store = SupabaseVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        client=supabase_client,
        table_name="documents",           # The 384-dimension table in your database
        query_name="match_documents",      # The matching RPC function
        chunk_size=200                     # Comfortable bulk uploading size for HF API responses
    )
    return vector_store