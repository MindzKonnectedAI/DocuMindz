from langchain_cohere import CohereRerank
from cohere.client import Client as CohereClient
import os

# Cohere setup (for reranking)
cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_client = CohereClient(api_key=cohere_api_key)

cohere_reranker = CohereRerank(model="rerank-english-v3.0",client=cohere_client)