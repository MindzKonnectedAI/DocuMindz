import os
from langchain_openai import ChatOpenAI # OpenAI's Chat Model
from dotenv import load_dotenv  # .env file loading
load_dotenv(override=True)

# OpenAI setup
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    api_key=openai_api_key,
    temperature=0,
    model="gpt-4o-mini",
    streaming=True,
    cache=False
)