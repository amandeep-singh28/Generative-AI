from dotenv import load_dotenv
import os
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

llm = ChatOpenAI(
    model = "openai/gpt-oss-20b",
    api_key = os.getenv("GROK_API_KEY"),
    base_url = "https://api.groq.com/openai/v1",
)

chat_history = [
    SystemMessage(content = "You are a helpful AI assistant")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content = user_input))
    if user_input == "exit":
        break
    result = llm.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print("AI: ", result.content)
print(chat_history)