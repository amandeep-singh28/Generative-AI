from dotenv import load_dotenv
import os
load_dotenv()

from langchain_openai import ChatOpenAI
from typing import TypedDict

llm = ChatOpenAI(
    model = "openai/gpt-oss-20b",
    api_key = os.getenv("GROK_API_KEY"),
    base_url = "https://api.groq.com/openai/v1",
)

# schema
class Review(TypedDict):
    summary : str
    sentiment : str

structured_model = llm.with_structured_output(Review)

result = structured_model.invoke(
    "The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this"
)

print(result)
print(result['summary'])
print(result['sentiment'])