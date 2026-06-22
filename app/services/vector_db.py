import os
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

supabase_client: Client = (
    create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    if SUPABASE_URL and SUPABASE_SERVICE_KEY
    else None
)

# BGE-large-en-v1.5 = 1024-dim vectors
# Supabase table must be: embedding vector(1024)
embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-base-en-v1.5",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)

def get_vector_store() -> SupabaseVectorStore | None:
    """Returns the vector store instance for use in retrieval chains."""
    if not supabase_client:
        print("Error: Supabase client not initialized. Check environment variables.")
        return None

    return SupabaseVectorStore(
        embedding=embeddings,
        client=supabase_client,
        table_name="documents",
        query_name="match_documents"
    )

def get_retriever(k: int = 5, source_filter: str | None = None):
    """
    Returns a LangChain retriever.
    
    Args:
        k: number of chunks to retrieve
        source_filter: optionally filter by metadata source tag
                       e.g. "events", "profiles", "alumni"
    """
    vector_store = get_vector_store()
    if not vector_store:
        return None

    search_kwargs = {"k": k}
    if source_filter:
        search_kwargs["filter"] = {"source": source_filter}

    return vector_store.as_retriever(search_kwargs=search_kwargs)