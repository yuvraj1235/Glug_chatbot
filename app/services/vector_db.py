import os
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Replace the local disk path with your Supabase Cloud credentials
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") # Use the service_role key to bypass RLS for inserts

# Initialize the official Supabase client
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Keeping your fast, free embedding model exactly the same
# Note: 'all-MiniLM-L6-v2' creates vectors with 384 dimensions. 
# Ensure your Supabase table column is set to `vector(384)`!
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def store_documents_in_supabase(documents):
    # 2. Swap Chroma for SupabaseVectorStore
    vector_store = SupabaseVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        client=supabase_client,
        table_name="documents",           # The table name we created in your database
        query_name="match_documents"       # The matching RPC function we created
    )
    return vector_store