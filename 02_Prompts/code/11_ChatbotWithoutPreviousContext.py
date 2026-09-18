from dotenv import load_dotenv
import os
load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model = "openai/gpt-oss-20b",
    api_key = os.getenv("GROK_API_KEY"),
    base_url = "https://api.groq.com/openai/v1",
)

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    result = llm.invoke(user_input)
    print("AI: ", result.content)
