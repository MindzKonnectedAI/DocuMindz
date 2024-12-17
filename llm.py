from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv  # .env file loading
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
llmmodel = ChatOpenAI(
    api_key=openai_api_key,
    temperature=0,
    model="gpt-4o-mini",
    streaming=True,
)