import os
import chromadb
from chromadb.utils import embedding_functions

# Use Cloudflare's Text Embedding model to keep embeddings consistent with the LLM
class CloudflareEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, account_id: str, api_token: str):
        self.account_id = account_id
        self.api_token = api_token
        # Standard OpenAI client layout to talk to CF embedding model
        from openai import OpenAI
        self.client = OpenAI(
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
            api_key=api_token
        )

    def __call__(self, input: list[str]) -> list[list[float]]:
        # Call Cloudflare's verified BAAI embedding model primitive
        response = self.client.embeddings.create(
            model="@cf/baai/bge-large-en-v1.5",
            input=input
        )
        return [item.embedding for item in response.data]

# Path to persist vector files locally
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db_storage")

def get_vector_collection():
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")

    # Initialize persistent local storage disk client
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    embedding_fn = CloudflareEmbeddingFunction(account_id, api_token)

    # Get or create a collection for Canonical Docs
    collection = chroma_client.get_or_create_collection(
        name="canonical_product_docs",
        embedding_function=embedding_fn
    )
    return collection
