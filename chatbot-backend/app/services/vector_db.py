import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# This will create a 'chroma_data' folder in your project root to store the DB
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../../chroma_data")

# Using a fast, free, local open-source embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def store_documents_in_chroma(documents):
    # This creates the vector embeddings and saves them to the disk
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vector_store