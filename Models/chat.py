import os
from langchain_openai import ChatOpenAI # OpenAI's Chat Model

# OpenAI setup
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    api_key=openai_api_key,
    temperature=0,
    model="gpt-4o-mini",
    streaming=True,
)