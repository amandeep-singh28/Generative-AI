from dotenv import load_dotenv
import os
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model = "openai/gpt-oss-20b",
    api_key = os.getenv("GROK_API_KEY"),
    base_url = "https://api.groq.com/openai/v1",
)

# chat_template
chat_template = ChatPromptTemplate([
    ("system", "You are a helpful customer support system"),
    MessagesPlaceholder(variable_name = "chat_history"),
    ("human", "{query}")
])

# load chat history
chat_history = []
with open(r"D:\Generative AI\Files\chat_history.txt") as f:
    chat_history.extend(f.readlines())
print(chat_history)

# create prompt

prompt = chat_template.invoke({
    "chat_history" : chat_history,
    "query" : "Where is my refund"
})

response = llm.invoke(prompt)
print(response.content)
