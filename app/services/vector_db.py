import os
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
# 1. FIXED: Swapped to local CPU execution wrapper to bypass HF cloud limits completely
from langchain_community.embeddings import HuggingFaceEmbeddings

# Replace the local disk path with your Supabase Cloud credentials
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") # Use the service_role key to bypass RLS for inserts

# Initialize the official Supabase client
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) if SUPABASE_URL and SUPABASE_SERVICE_KEY else None

# 2. FIXED: Local model execution. This downloads the model weights (~100MB) once 
# on the first run and runs calculations completely offline. 
# Matches your 384-dimensional vector database table perfectly!
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def store_documents_in_supabase(documents):
    if not supabase_client:
        print("Error: Supabase client is not initialized. Check your environment variables.")
        return None
        
    # Swap Chroma for SupabaseVectorStore
    vector_store = SupabaseVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        client=supabase_client,
        table_name="documents",           # The 384-dimension table in your database
        query_name="match_documents",      # The matching RPC function
        chunk_size=100                     # Comfortable batch size for inserting records into Supabase
    )
    return vector_store